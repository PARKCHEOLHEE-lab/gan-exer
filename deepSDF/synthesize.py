import torch

from tqdm import tqdm
from typing import List
from deepSDF.model import SDFdecoder
from deepSDF.config import Configuration
from deepSDF.reconstruct import ReconstructorHelper


class SynthesizerHelper:
    @staticmethod
    def interpolate(latent_codes: List[torch.Tensor], factors: List[float]) -> torch.Tensor:
        """Interpolate latent codes.

        Args:
            latent_codes (List[torch.Tensor]): latent codes
            factors (List[float]): factors to interpolate

        Returns:
            torch.Tensor: interpolated latent code
        """

        assert len(latent_codes) - 1 == len(factors), "Given inputs are not valid for interpolation."

        if len(latent_codes) == 1:
            return latent_codes[0]

        interpolated = latent_codes[0]
        for fi, f in enumerate(factors):
            interpolated = interpolated * (1 - f) + latent_codes[fi + 1] * f

        return interpolated


class Synthesizer(ReconstructorHelper, SynthesizerHelper, Configuration):
    def __init__(self) -> None:
        pass

    def synthesize(
        self,
        sdf_decoder: SDFdecoder,
        latent_code: torch.Tensor,
        save_name: str = None,
        resolution: int = Configuration.RECONSTRUCT_RESOLUTION,
        normalize: bool = True,
        map_z_to_y: bool = False,
        check_watertight: bool = False,
    ):
        """Synthesize skyscrapers

        cls_dict = {
            0: 'bank_of_china',
            1: 'burj_al_arab',
            2: 'cctv_headquarter',
            3: 'china_zun',
            4: 'empire_state_building',
            5: 'hearst_tower',
            6: 'kingdom_centre',
            7: 'lotte_tower',
            8: 'mahanakhon',
            9: 'one_world_trade_center',
            10: 'shanghai_world_financial_center',
            11: 'taipei_101',
            12: 'the_gherkin',
            13: 'the_shard',
            14: 'transamerica_pyramid'
        }

        Args:
            sdf_decoder (SDFdecoder): model
            latent_code (torch.Tensor): latent code
            save_name (str): save name
            resolution (int, optional): resolution for reconstruction. Defaults to Configuration.RECONSTRUCT_RESOLUTION.
            normalize (bool, optional): normalize. Defaults to True.
            map_z_to_y (bool, optional): map z to y. Defaults to False.
        """

        sdf_decoder.eval()

        with torch.no_grad():
            coords, grid_size_axis = self.get_volume_coords(resolution=resolution)
            coords.to(self.DEVICE)
            coords_batches = torch.split(coords, coords.shape[0] // 1000)

            sdf = torch.tensor([]).to(self.DEVICE)

            for coords_batch in tqdm(coords_batches, desc="Synthesizing ... ", leave=False):
                interpolated_repeat = latent_code.unsqueeze(1).repeat(1, coords_batch.shape[0]).transpose(0, 1)
                cxyz_1 = torch.cat([interpolated_repeat, coords_batch], dim=1)
                pred = sdf_decoder(None, None, cxyz_1)

                if sum(sdf.shape) == 0:
                    sdf = pred
                else:
                    sdf = torch.vstack([sdf, pred])

            mesh = self.extract_mesh(
                grid_size_axis=grid_size_axis,
                sdf=sdf,
                normalize=normalize,
                map_z_to_y=map_z_to_y,
                check_watertight=check_watertight,
            )

            if mesh is not None and save_name is not None:
                mesh.export(save_name)

        sdf_decoder.train()

        return mesh
