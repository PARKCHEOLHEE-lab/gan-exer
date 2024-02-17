import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple
from lirGAN.config import ModelConfig
from lirGAN.data.data_creator import DataCreator
from lirGAN.data import utils
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler
from IPython.display import clear_output


class LirDataset(Dataset, ModelConfig):
    def __init__(self, data_path: str = None, slicer: int = np.inf):
        self.lir_dataset: List[torch.Tensor]
        self.lir_dataset = self._get_lir_dataset(
            data_path=self.DATA_PATH if data_path is None else data_path, slicer=slicer
        )

    def __len__(self) -> int:
        return len(self.lir_dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        input_polygon, target_lir = self.lir_dataset[index]

        return input_polygon.unsqueeze(dim=0).to(self.DEVICE), target_lir.unsqueeze(dim=0).to(self.DEVICE)

    def _get_lir_dataset(self, data_path: str, slicer: int) -> List[torch.Tensor]:
        """Load the lir dataset

        Args:
            data_path (str): path to load data

        Returns:
            List[torch.Tensor]: Each tensor is (2, 500, 500)-shaped tensor.
                                The first one is the input polygon and the second one is ground-truth rectangle
        """

        lir_dataset: List[torch.Tensor]
        lir_dataset = []

        for file_name in os.listdir(data_path):
            lir_data_path = os.path.join(data_path, file_name)
            lir_dataset.append(torch.tensor(np.load(lir_data_path), dtype=torch.float))

        if slicer < np.inf:
            return lir_dataset[:slicer]

        return lir_dataset


class LirGeometricLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        diou_weight: float = 1.0,
        feasibility_weight: float = 1.0,
        connectivity_weight: float = 1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.diou_weight = diou_weight
        self.feasibility_weight = feasibility_weight
        self.connectivity_weight = connectivity_weight

        self.bce_loss_function = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_diou_loss(generated_lir: torch.Tensor, target_lir: torch.Tensor, diou_weight: float) -> torch.Tensor:
        """compute the distance-IoU loss

        Args:
            generated_lir (torch.Tensor):  generated rectangle
            target_lir (torch.Tensor): target rectangle
            diou_weight (float): weight to multiply in the diou loss

        Returns:
            torch.Tensor: diou loss
        """

        generated_lir = (generated_lir > 0.5).float()

        intersection = torch.logical_and(generated_lir, target_lir).float()
        union = torch.logical_or(generated_lir, target_lir).float()
        iou_score = torch.sum(intersection) / torch.sum(union)

        generated_lir_centroid = torch.nonzero(generated_lir).float().mean(dim=0)
        if torch.isnan(generated_lir_centroid).any():
            generated_lir_centroid = torch.zeros_like(generated_lir_centroid)

        target_lir_centroid = torch.nonzero(target_lir).float().mean(dim=0)

        distance = torch.norm(generated_lir_centroid - target_lir_centroid)

        diagonal = torch.norm(torch.tensor(generated_lir.shape).float())

        normalized_distance = (distance / diagonal) ** 2

        diou_loss = 1 - (iou_score - normalized_distance)

        return diou_loss * diou_weight

    @staticmethod
    def compute_feasibility_loss(
        polygon_to_check: torch.Tensor,
        generated_lir: torch.Tensor,
        feasibility_weight: float,
        with_underfitting: bool = False,
    ) -> torch.Tensor:
        """compute the feasibility loss that checks the generated rectangle is within the input polygon

        Args:
            input_polygon (torch.Tensor): input polygon boundary to predict
            generated_lir (torch.Tensor): generated rectangle
            feasibility_weight (float): weight to multiply in the feasibility loss

        Returns:
            torch.Tensor: feasibility loss
        """

        generated_lir = (generated_lir > 0.5).float()

        feasibility_loss = ((generated_lir == 1) & (polygon_to_check == 0)).sum().float()
        normalized_feasibility_loss = (feasibility_loss / polygon_to_check.sum().float()) ** 2

        normalized_underfitting = torch.zeros_like(normalized_feasibility_loss)
        if with_underfitting:
            underfitting = ((generated_lir == 0) & (polygon_to_check == 1)).sum().float()
            normalized_underfitting = (underfitting / polygon_to_check.sum().float()) ** 2

        return (normalized_feasibility_loss + normalized_underfitting) * feasibility_weight

    @staticmethod
    def compute_connectivity_loss(generated_lir: torch.Tensor, connectivity_weight: float) -> torch.Tensor:
        """compute the connectivity loss that checks whether the generated rectangle is a single piece

        Args:
            generated_lir (torch.Tensor): generated rectangle
            connectivity_weight (float): weight to multiply in the connectivity loss.

        Returns:
            torch.Tensor: connectivity loss
        """

        generated_lir_np = (generated_lir.detach().cpu().numpy() > 0.5).astype("uint8").squeeze()

        num_labels_gen, _ = cv2.connectedComponents(generated_lir_np, connectivity=4)
        num_labels_gen -= 1

        connectivity_loss = torch.tensor(abs(num_labels_gen - 1)).float().to(generated_lir.device)
        normalized_connectivity_loss = (connectivity_loss / generated_lir_np.size) ** 2

        return normalized_connectivity_loss * connectivity_weight

    def forward(
        self, input_polygons: torch.Tensor, generated_lirs: torch.Tensor, target_lirs: torch.Tensor
    ) -> torch.Tensor:
        """compute total loss

        Args:
            input_polygons (torch.Tensor): input polygon boundaries to predict
            generated_lirs (torch.Tensor): generated rectangles
            target_lirs (torch.Tensor): target rectangles

        Returns:
            torch.Tensor: loss merged with bce, diou, feasibility
        """

        total_loss = 0
        for generated_lir, target_lir, input_polygon in zip(generated_lirs, target_lirs, input_polygons):
            bce_loss = self.bce_loss_function(generated_lir, target_lir) * self.bce_weight

            diou_loss = LirGeometricLoss.compute_diou_loss(generated_lir, target_lir, self.diou_weight)

            connectivity_loss = LirGeometricLoss.compute_connectivity_loss(generated_lir, self.connectivity_weight)

            feasibility_loss_1 = LirGeometricLoss.compute_feasibility_loss(
                input_polygon, generated_lir, self.feasibility_weight
            )

            feasibility_loss_2 = LirGeometricLoss.compute_feasibility_loss(
                target_lir, generated_lir, self.feasibility_weight, with_underfitting=True
            )

            assert not torch.isnan(diou_loss).any(), "diou_loss is `nan`"
            assert not torch.isnan(feasibility_loss_1).any(), "feasibility_loss_1 is `nan`"
            assert not torch.isnan(feasibility_loss_2).any(), "feasibility_loss_2 is `nan`"
            assert not torch.isnan(connectivity_loss).any(), "connectivity_loss is `nan`"

            loss = bce_loss + diou_loss + connectivity_loss + feasibility_loss_1 + feasibility_loss_2

            total_loss += loss

        return total_loss


class WeightsInitializer:
    @staticmethod
    def initialize_weights_xavier(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def initialize_weights_he(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class LirGenerator(nn.Module, ModelConfig):
    def __init__(self, use_tanh: bool):
        super().__init__()

        self.use_tanh = use_tanh

        self.linear = nn.Sequential(
            nn.Linear(256 * 256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 1024, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
        )

        self.to(self.DEVICE)

    def forward(self, noise, input_polygon):
        fc = self.linear(input_polygon.reshape(input_polygon.shape[0], -1))
        x = torch.cat([noise, fc], dim=1)
        x = x.reshape(x.shape[0], 256, 1, 1)
        x = self.main(x)

        if self.use_tanh:
            return nn.Tanh()(x)

        return nn.Sigmoid()(x)


class LirDiscriminator(nn.Module, ModelConfig):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )

        self.to(self.DEVICE)

    def forward(self, rectangle, input_polygon):
        x = torch.cat([rectangle, input_polygon], dim=1)
        return self.main(x).view(-1, 1).squeeze(1)


class LirGanTrainer(ModelConfig, WeightsInitializer):
    def __init__(
        self,
        epochs: int,
        lir_generator: LirGenerator,
        lir_discriminator: LirDiscriminator,
        lir_dataloader: DataLoader,
        lir_geometric_loss_function: LirGeometricLoss = None,
        lir_discriminator_loss_function: _Loss = nn.BCEWithLogitsLoss(),
        initial_weights_key: str = None,
        log_interval: int = None,
        record_name: str = None,
        use_gradient_penalty: bool = False,
        use_lr_scheduler: bool = False,
        is_debug_mode: bool = False,
        is_record: bool = False,
    ):
        self.epochs = epochs
        self.lir_generator = lir_generator
        self.lir_discriminator = lir_discriminator
        self.lir_dataloader = lir_dataloader
        self.lir_geometric_loss_function = lir_geometric_loss_function
        self.lir_discriminator_loss_function = lir_discriminator_loss_function
        self.initial_weights_key = initial_weights_key
        self.log_interval = self.LOG_INTERVAL if log_interval is None else log_interval
        self.record_name = record_name
        self.use_gradient_penalty = use_gradient_penalty
        self.use_lr_scheduler = use_lr_scheduler
        self.is_debug_mode = is_debug_mode
        self.is_record = is_record

        self._make_dirs_and_assign_paths()
        self._set_latest_pths()
        self._set_initial_weights()
        self._set_optimizers()
        self._set_lr_schedulers()

    def _make_dirs_and_assign_paths(self) -> None:
        """Make directories to record"""

        if self.record_name is not None and self.is_record:
            self.records_path = os.path.abspath(os.path.join(__file__, "../", "records"))
            if not os.path.isdir(self.records_path):
                os.mkdir(self.records_path)

            self.records_path_with_name = os.path.join(self.records_path, self.record_name)
            if not os.path.isdir(self.records_path_with_name):
                os.mkdir(self.records_path_with_name)

            self.records_path_polygons = os.path.join(self.records_path_with_name, "polygons")
            if not os.path.isdir(self.records_path_polygons):
                os.mkdir(self.records_path_polygons)

            self.records_path_grpahs = os.path.join(self.records_path_with_name, "graphs")
            if not os.path.isdir(self.records_path_grpahs):
                os.mkdir(self.records_path_grpahs)

            self.records_path_losses = os.path.join(self.records_path_with_name, "losses")
            if not os.path.isdir(self.records_path_losses):
                os.mkdir(self.records_path_losses)

            self.pths_path = os.path.abspath(os.path.join(__file__, "../", "pths"))
            if not os.path.isdir(self.pths_path):
                os.mkdir(self.pths_path)

            self.pths_path_with_name = os.path.join(self.pths_path, self.record_name)
            if not os.path.isdir(self.pths_path_with_name):
                os.mkdir(self.pths_path_with_name)

            self.lir_generator_pth_path = os.path.join(self.pths_path_with_name, "lir_generator.pth")
            self.lir_discriminator_pth_path = os.path.join(self.pths_path_with_name, "lir_discriminator.pth")

    def _set_latest_pths(self) -> None:
        """Set path to save checkpoint of models"""

        self.is_pths_set = False

        if (
            self.record_name is not None
            and os.path.isfile(self.lir_generator_pth_path)
            and os.path.isfile(self.lir_discriminator_pth_path)
        ):
            self.lir_generator.load_state_dict(
                torch.load(self.lir_generator_pth_path, map_location=torch.device(self.DEVICE))
            )

            self.lir_discriminator.load_state_dict(
                torch.load(self.lir_discriminator_pth_path, map_location=torch.device(self.DEVICE))
            )

            print("Set initial weights from existing .pths:")
            print(f"  generator_pth_path:     {self.lir_generator_pth_path}")
            print(f"  discriminator_pth_path: {self.lir_discriminator_pth_path}")
            print()

            self.is_pths_set = True

    def _set_lr_schedulers(self) -> None:
        """Set each scheduler for optimizers"""

        if self.use_lr_scheduler:
            self.scheduler_g = lr_scheduler.ReduceLROnPlateau(
                self.lir_generator_optimizer, patience=3, verbose=True, factor=0.1
            )
            self.scheduler_d = lr_scheduler.ReduceLROnPlateau(
                self.lir_discriminator_optimizer, patience=3, verbose=True, factor=0.1
            )

    def _set_optimizers(self) -> None:
        """Set each optimizer for models"""

        self.lir_generator_optimizer = torch.optim.Adam(
            self.lir_generator.parameters(), lr=self.LEARNING_RATE, betas=self.BETAS
        )
        self.lir_discriminator_optimizer = torch.optim.Adam(
            self.lir_discriminator.parameters(), lr=self.LEARNING_RATE, betas=self.BETAS
        )

    def _set_initial_weights(self) -> None:
        """Set initial weights by xavier or he keys"""

        if not self.is_pths_set and self.initial_weights_key is not None:
            if self.initial_weights_key == self.XAVIER:
                self.lir_generator.apply(self.initialize_weights_xavier)
                self.lir_discriminator.apply(self.initialize_weights_xavier)
            elif self.initial_weights_key == self.HE:
                self.lir_generator.apply(self.initialize_weights_he)
                self.lir_discriminator.apply(self.initialize_weights_he)

    def _compute_gradient_penalty(
        self, target_lirs: torch.Tensor, generated_lirs: torch.Tensor, input_polygons: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient penalty to enforce the Lipschitz constraint.

        Args:
            target_lirs (torch.Tensor): target rectangles
            generated_lirs (torch.Tensor): generated rectangles
            input_polygons (torch.Tensor): input polygon boundaries to predict

        Returns:
            torch.Tensor: The computed gradient penalty.
        """

        batch_size, *_ = target_lirs.size()
        alpha = torch.rand((batch_size, 1, 1, 1)).to(self.DEVICE)

        interpolated = (alpha * target_lirs + ((1 - alpha) * generated_lirs)).requires_grad_(True).to(self.DEVICE)

        d_interpolated = self.lir_discriminator(interpolated, input_polygons)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(self.DEVICE),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        gradient_penalty = self.LAMBDA * ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty

    def _get_noise(self, size: Tuple[int]) -> torch.Tensor:
        """Returns the noise corresponding to the given size.

        Args:
            size (Tuple[int]): noise shape

        Returns:
            torch.Tensor: noise
        """

        return torch.randn(size).to(self.DEVICE)

    def _get_smoothed_labels(self, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """Applies label smoothing to reduce model overconfidence.

        Args:
            labels (torch.Tensor): Original labels.
            smoothing (float): Smoothing factor, default is 0.1.

        Returns:
            torch.Tensor: Smoothed labels.
        """

        return labels * (1 - smoothing) + 0.5 * smoothing

    def _train_lir_discriminator(self, input_polygons: torch.Tensor, target_lirs: torch.Tensor) -> torch.Tensor:
        """Trains the discriminator

        Args:
            input_polygons (torch.Tensor): input polygon boundaries to predict
            target_lirs (torch.Tensor): target rectangles

        Returns:
            torch.Tensor: discriminator's loss
        """

        noise = self._get_noise((input_polygons.shape[0], self.NOISE_DIM))
        generated_lirs = self.lir_generator(noise, input_polygons)

        real_d = self.lir_discriminator(target_lirs, input_polygons)
        fake_d = self.lir_discriminator(generated_lirs, input_polygons)

        loss_real_d = self.lir_discriminator_loss_function(real_d, self._get_smoothed_labels(torch.ones_like(real_d)))
        loss_fake_d = self.lir_discriminator_loss_function(fake_d, self._get_smoothed_labels(torch.zeros_like(fake_d)))

        loss_d = loss_real_d + loss_fake_d
        if self.use_gradient_penalty:
            loss_d += self._compute_gradient_penalty(target_lirs, generated_lirs, input_polygons)

        self.lir_discriminator_optimizer.zero_grad()
        loss_d.backward()
        self.lir_discriminator_optimizer.step()

        return loss_d

    def _train_lir_generator(self, input_polygons: torch.Tensor, target_lirs: torch.Tensor) -> torch.Tensor:
        """Trains the generator

        Args:
            input_polygons (torch.Tensor): input polygon boundaries to predict
            target_lirs (torch.Tensor): target rectangles

        Returns:
            torch.Tensor: generator's loss
        """

        noise = self._get_noise((input_polygons.shape[0], self.NOISE_DIM))
        generated_lir = self.lir_generator(noise, input_polygons)

        fake_d = self.lir_discriminator(generated_lir, input_polygons)

        adversarial_loss = self.lir_discriminator_loss_function(
            fake_d, self._get_smoothed_labels(torch.ones_like(fake_d))
        )

        geometric_loss = 0
        if self.lir_geometric_loss_function is not None:
            geometric_loss = self.lir_geometric_loss_function(input_polygons, generated_lir, target_lirs)

        loss_g = adversarial_loss + geometric_loss

        self.lir_generator_optimizer.zero_grad()
        loss_g.backward()
        self.lir_generator_optimizer.step()

        return loss_g

    def evaluate(
        self,
        batch_size_to_evaulate: int,
        batch_size_to_evaulate_trained_data: int,
        input_polygons: torch.Tensor,
        target_lirs: torch.Tensor,
        losses_g: List[torch.Tensor],
        losses_d: List[torch.Tensor],
        polygons_save_path: str,
        graphs_save_path: str,
    ) -> None:
        """Evaluates and shows how well the model works by creating pictures and graphs.

        Args:
            batch_size_to_evaulate (int): size to evaluate and visualize
            batch_size_to_evaulate_trained_data (int): size to evaluate and visualize
            input_polygons (torch.Tensor): input polygon boundaries to predict
            target_lirs (torch.Tensor): target rectangles
            losses_g (List[torch.Tensor]): average losses of generator
            losses_d (List[torch.Tensor]): average losses of discriminator
            polygons_save_path (str): path to save generated polygon figure
            graphs_save_path (str): path to save losses figure
        """

        self.lir_generator.eval()
        self.lir_discriminator.eval()

        with torch.no_grad():
            binary_grids = []
            if len(self.lir_dataloader) > 1:
                data_creator = DataCreator(creation_count=None)
                for _ in range(batch_size_to_evaulate):
                    random_coordinates = data_creator._get_fitted_coordinates(
                        coordinates=data_creator._get_random_coordinates(
                            vertices_count_min=data_creator.random_vertices_count_min,
                            vertices_count_max=data_creator.random_vertices_count_max,
                        ),
                        canvas_size=data_creator.canvas_size,
                    )

                    binary_grid_shaped_polygon = utils.get_binary_grid_shaped_polygon(
                        random_coordinates.astype(np.int32), data_creator.canvas_size
                    )

                    random_input_polygon = (
                        torch.tensor(binary_grid_shaped_polygon, dtype=torch.float)
                        .unsqueeze(dim=0)
                        .unsqueeze(dim=0)
                        .to(self.DEVICE)
                    )

                    noise = self._get_noise((1, self.NOISE_DIM))
                    generated_lir = self.lir_generator(noise, random_input_polygon) > 0.5
                    generated_lir = generated_lir.squeeze().detach().cpu().numpy().astype(int)
                    generated_lir = np.logical_and(generated_lir, binary_grid_shaped_polygon)

                    merged = binary_grid_shaped_polygon + generated_lir
                    binary_grids.append(merged)

                for input_polygon in input_polygons[:batch_size_to_evaulate_trained_data]:
                    noise = self._get_noise((1, self.NOISE_DIM))
                    generated_lir = self.lir_generator(noise, input_polygon) > 0.5
                    generated_lir = generated_lir.squeeze().detach().cpu().numpy().astype(int)

                    input_polygon = input_polygon.squeeze().detach().cpu().numpy().astype(int)

                    generated_lir = np.logical_and(generated_lir, input_polygon)

                    merged = input_polygon + generated_lir
                    binary_grids.append(merged)

            else:
                noise = self._get_noise((input_polygons.shape[0], self.NOISE_DIM))
                input_polygon = input_polygons.squeeze().detach().cpu().numpy()
                target_lir = target_lirs.squeeze().detach().cpu().numpy()

                generated_lir = (
                    (self.lir_generator(noise, input_polygons) > 0.5).squeeze().detach().cpu().numpy().astype(int)
                )

                generated_lir = np.logical_and(generated_lir, input_polygon)

                binary_grids.extend([input_polygon + target_lir, input_polygon + generated_lir])

            utils.plot_binary_grids(binary_grids, save_path=polygons_save_path)
            utils.plot_losses(losses_g=losses_g, losses_d=losses_d, save_path=graphs_save_path)

        self.lir_generator.train()
        self.lir_discriminator.train()

    def train(self) -> None:
        """Main function to train models"""

        if self.is_debug_mode:
            from debugvisualizer.debugvisualizer import Plotter
            from shapely.geometry import Point, Polygon

            globals()["Plotter"] = Plotter
            globals()["Point"] = Point
            globals()["Polygon"] = Polygon

        losses_g = []
        losses_d = []

        polygons_save_path = None
        graphs_save_path = None

        if self.record_name is not None and self.is_record:
            losses_npy_path = os.path.join(self.records_path_losses, "losses.npy")
            if os.path.isfile(losses_npy_path):
                losses = np.load(losses_npy_path)
                losses_g = list(losses[0])
                losses_d = list(losses[1])

        epoch_addition = len(losses_g)

        for epoch in range(1, self.epochs + 1):
            epoch += epoch_addition

            if epoch % self.log_interval == 0:
                clear_output(wait=True)

            losses_g_per_epoch = []
            losses_d_per_epoch = []

            for input_polygons, target_lirs in self.lir_dataloader:
                loss_d = self._train_lir_discriminator(input_polygons, target_lirs)
                loss_g = self._train_lir_generator(input_polygons, target_lirs)
                losses_g_per_epoch.append(loss_g.item())
                losses_d_per_epoch.append(loss_d.item())

                self.evaluate(
                    batch_size_to_evaulate=3,
                    batch_size_to_evaulate_trained_data=3,
                    input_polygons=input_polygons,
                    target_lirs=target_lirs,
                    losses_g=losses_g_per_epoch,
                    losses_d=losses_d_per_epoch,
                    polygons_save_path=None,
                    graphs_save_path=None,
                )

                clear_output(wait=True)

            avg_loss_g = np.mean(losses_g_per_epoch)
            avg_loss_d = np.mean(losses_d_per_epoch)
            losses_g.append(avg_loss_g)
            losses_d.append(avg_loss_d)

            if self.use_lr_scheduler:
                self.scheduler_g.step(avg_loss_g)
                self.scheduler_d.step(avg_loss_d)

            if epoch % self.log_interval == 0:
                if self.record_name is not None and self.is_record:
                    polygons_save_path = os.path.join(self.records_path_polygons, f"polygons-{epoch}.png")
                    graphs_save_path = os.path.join(self.records_path_grpahs, f"graphs-{epoch}.png")

                self.evaluate(
                    batch_size_to_evaulate=3,
                    batch_size_to_evaulate_trained_data=3,
                    input_polygons=input_polygons,
                    target_lirs=target_lirs,
                    losses_g=losses_g,
                    losses_d=losses_d,
                    polygons_save_path=polygons_save_path,
                    graphs_save_path=graphs_save_path,
                )

                if self.record_name is not None and self.is_record:
                    np.save(losses_npy_path, np.array([losses_g, losses_d]))
                    torch.save(self.lir_generator.state_dict(), self.lir_generator_pth_path)
                    torch.save(self.lir_discriminator.state_dict(), self.lir_discriminator_pth_path)
