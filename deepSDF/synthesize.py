import os
import time
import torch
import random
import commonutils
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from deepSDF.model import SDFdecoder
from IPython.display import clear_output
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

    def random_interpolation_synthesis(
        self, sdf_decoder: SDFdecoder, latent_codes_data: dict
    ) -> Tuple[int, int, float, torch.Tensor]:
        """Randomly synthesize latent codes by interpolating them

        Args:
            sdf_decoder (SDFdecoder): model
            latent_codes (dict): all latent codes

        Returns:
            Tuple[int, int, float, torch.Tensor]: selected indices, interpolation factor, and synthesized latent code
        """

        data_to_sample = latent_codes_data["data"]

        if random.Random(time.time()).random() < 0.5:
            data_to_sample = data_to_sample[: len(sdf_decoder.latent_codes)]

        data_to_sample = [rd for rd in data_to_sample if rd["synthesis_type"] != "arithmetic"]

        data_1, data_2 = random.Random(time.time()).sample(data_to_sample, 2)

        latent_code_1 = data_1["latent_code"]
        latent_code_1 = torch.tensor(latent_code_1).to(sdf_decoder.latent_codes.device)
        latent_code_1_index = data_1["index"]

        latent_code_2 = data_2["latent_code"]
        latent_code_2 = torch.tensor(latent_code_2).to(sdf_decoder.latent_codes.device)
        latent_code_2_index = data_2["index"]

        selected_indices = f"{latent_code_1_index}__{latent_code_2_index}"

        random_interpolation_factor = round(0.25 + (0.75 - 0.25) * random.Random(time.time()).random(), 3)

        synthesized_latent_code = self.interpolate(
            latent_codes=[latent_code_1, latent_code_2], factors=[random_interpolation_factor]
        )

        return selected_indices, random_interpolation_factor, synthesized_latent_code

    def random_arithmetic_operations_synthesis(
        self, sdf_decoder: SDFdecoder, latent_codes_data: dict
    ) -> Tuple[str, torch.Tensor]:
        """Randomly synthesize latent codes by arithmetic operations

        Args:
            sdf_decoder (SDFdecoder): model
            latent_codes_data (dict): all latent codes

        Returns:
            Tuple[str, torch.Tensor]: selected indices and synthesized latent code
        """

        data_to_sample = latent_codes_data["data"]

        if random.Random(time.time()).random() < 0.5:
            data_to_sample = data_to_sample[: len(sdf_decoder.latent_codes)]

        data_to_sample = [rd for rd in data_to_sample if rd["synthesis_type"] != "interpolation"]

        random_data = random.Random(time.time()).sample(data_to_sample, 3)

        selected_indices = str(random_data[0]["index"])
        synthesized_latent_code = torch.tensor(random_data[0]["latent_code"]).to(sdf_decoder.latent_codes.device)
        for rdi, rd in enumerate(random_data[1:]):
            if rdi != len(random_data[1:]) - 1:
                synthesized_latent_code += torch.tensor(rd["latent_code"]).to(sdf_decoder.latent_codes.device)
            else:
                synthesized_latent_code -= torch.tensor(rd["latent_code"]).to(sdf_decoder.latent_codes.device)

            selected_indices += "__" + str(rd["index"])

        return selected_indices, synthesized_latent_code

    @torch.inference_mode()
    @commonutils.runtime_calculator
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

        return mesh


def infinite_synthesis(
    sdf_decoder: SDFdecoder,
    save_dir: str,
    synthesis_count: int = np.inf,
    resolution: int = 128,
    map_z_to_y: bool = True,
    check_watertight: bool = True,
):
    synthesizer = Synthesizer()

    synthesized_latent_codes_npz = "infinite_synthesized_latent_codes.npz"
    synthesized_latent_codes_path = os.path.join(save_dir, synthesized_latent_codes_npz)

    os.makedirs(save_dir, exist_ok=True)

    synthesized_latent_codes = {
        "data": [
            {
                "name": i,
                "index": i,
                "synthesis_type": "initial",
                "latent_code": list(latent_code.detach().cpu().numpy()),
            }
            for i, latent_code in enumerate(sdf_decoder.latent_codes)
        ]
    }

    if os.path.exists(synthesized_latent_codes_path):
        synthesized_latent_codes = {
            "data": list(np.load(synthesized_latent_codes_path, allow_pickle=True)["synthesized_data"])
        }

    c = 0
    while c < synthesis_count:
        print("synthesized data length:", len(synthesized_latent_codes["data"]))

        if random.Random(time.time()).random() < 0.5:
            selected_indices, synthesized_latent_code = synthesizer.random_arithmetic_operations_synthesis(
                sdf_decoder=sdf_decoder, latent_codes_data=synthesized_latent_codes
            )

            synthesis_type = "arithmetic"

            name = f"{selected_indices}.obj"
            save_name = os.path.join(save_dir, name)

        else:
            (
                selected_indices,
                random_interpolation_factor,
                synthesized_latent_code,
            ) = synthesizer.random_interpolation_synthesis(
                sdf_decoder=sdf_decoder, latent_codes_data=synthesized_latent_codes
            )

            synthesis_type = "interpolation"

            name = f"{selected_indices}__{str(random_interpolation_factor).replace('.', '-')}.obj"
            save_name = os.path.join(save_dir, name)

        if os.path.exists(save_name):
            continue

        _ = synthesizer.synthesize(
            sdf_decoder=sdf_decoder,
            latent_code=synthesized_latent_code,
            resolution=resolution,
            save_name=save_name,
            map_z_to_y=map_z_to_y,
            check_watertight=check_watertight,
        )

        synthesized_data = {
            "name": name,
            "index": len(synthesized_latent_codes["data"]),
            "synthesis_type": synthesis_type,
            "latent_code": list(synthesized_latent_code.detach().cpu().numpy()),
        }

        synthesized_latent_codes["data"].append(synthesized_data)

        np.savez(
            synthesized_latent_codes_path,
            synthesized_data=np.array(synthesized_latent_codes["data"]),
        )

        clear_output(wait=False)

        c += 1
