from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np
import torch.nn as nn

from external.adaptors.fastreid_adaptor import FastReID

class OSNetReID(nn.Module):
    def __init__(self, model_name='osnet_ain_x1_0', embedding_dim=256, pretrained=True):
        super(OSNetReID, self).__init__()
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=pretrained
        )
        self.model.classifier = nn.Identity()
        for name, param in self.model.named_parameters():
            if 'conv4' not in name:
                param.requires_grad = False
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

class EnsembleOSNetReID(nn.Module):
    def __init__(self, model_osnet, model_sbs_s50, weight_osnet=0.5, weight_sbs_s50=0.5, embedding_dim=256):
        super(EnsembleOSNetReID, self).__init__()
        self.model_osnet = model_osnet
        self.model_sbs_s50 = model_sbs_s50
        self.weight_osnet = weight_osnet
        self.weight_sbs_s50 = weight_sbs_s50
        # Add projection layer for SBS_S50 to match OSNet's embedding dimension
        self.sbs_s50_projection = nn.Linear(2048, embedding_dim) if model_sbs_s50 else None

    def forward(self, x):
        emb_osnet = self.model_osnet(x) if self.model_osnet else torch.zeros(x.size(0), 256, device=x.device, dtype=torch.float32)
        emb_sbs_s50 = self.model_sbs_s50(x) if self.model_sbs_s50 else torch.zeros(x.size(0), 2048, device=x.device, dtype=torch.float32)
        if self.sbs_s50_projection and emb_sbs_s50.size(1) == 2048:
            # Cast to float32 to match projection layer's dtype
            emb_sbs_s50 = emb_sbs_s50.float()
            emb_sbs_s50 = self.sbs_s50_projection(emb_sbs_s50)
            emb_sbs_s50 = nn.functional.normalize(emb_sbs_s50, p=2, dim=1)
        return nn.functional.normalize(
            self.weight_osnet * emb_osnet + self.weight_sbs_s50 * emb_sbs_s50, p=2, dim=1
        )

class EmbeddingComputer:
    def __init__(
        self,
        dataset,
        test_dataset,
        grid_off,
        max_batch=1024,
        reid_model_type="osnet",
        reid_path_osnet=None,
        reid_path_sbs_s50=None,
        reid_weight_osnet=0.5,
        reid_weight_sbs_s50=0.5
    ):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch
        self.reid_model_type = reid_model_type.lower()
        self.reid_path_osnet = reid_path_osnet
        self.reid_path_sbs_s50 = reid_path_sbs_s50
        self.reid_weight_osnet = reid_weight_osnet
        self.reid_weight_sbs_s50 = reid_weight_sbs_s50
        self.normalize = False

        if self.reid_model_type not in ["osnet", "sbs_s50", "ensemble"]:
            raise ValueError("reid_model_type must be 'osnet', 'sbs_s50', or 'ensemble'")
        if self.reid_model_type == "osnet" and self.reid_path_osnet and not os.path.exists(self.reid_path_osnet):
            raise FileNotFoundError(f"OSNet model path {self.reid_path_osnet} does not exist")
        if self.reid_model_type == "sbs_s50" and self.reid_path_sbs_s50 and not os.path.exists(self.reid_path_sbs_s50):
            raise FileNotFoundError(f"SBS_S50 model path {self.reid_path_sbs_s50} does not exist")
        if self.reid_model_type == "ensemble":
            if self.reid_path_osnet and not os.path.exists(self.reid_path_osnet):
                raise FileNotFoundError(f"OSNet model path {self.reid_path_osnet} does not exist")
            if self.reid_path_sbs_s50 and not os.path.exists(self.reid_path_sbs_s50):
                raise FileNotFoundError(f"SBS_S50 model path {self.reid_path_sbs_s50} does not exist")
            if not (0 <= self.reid_weight_osnet <= 1 and 0 <= self.reid_weight_sbs_s50 <= 1):
                raise ValueError("Model weights must be between 0 and 1")
            if abs(self.reid_weight_osnet + self.reid_weight_sbs_s50 - 1.0) > 1e-6:
                raise ValueError("Sum of model weights must equal 1")

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int32)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1]:patch_coords[3], patch_coords[0]:patch_coords[2], :]
                if viz:
                    dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1,
                    )
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1]:patch_coords[3], patch_coords[0]:patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, self.crop_size)
                patches.append(patch)

        return torch.cat(patches, dim=0)

    def compute_embedding(self, img, bbox, tag):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            self.initialize_model()

        crops = []
        if self.grid_off:
            h, w = img.shape[:2]
            results = np.round(bbox).astype(np.int32)
            results[:, 0] = results[:, 0].clip(0, w)
            results[:, 1] = results[:, 1].clip(0, h)
            results[:, 2] = results[:, 2].clip(0, w)
            results[:, 3] = results[:, 3].clip(0, h)

            for p in results:
                crop = img[p[1]:p[3], p[0]:p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)
        crops = torch.cat(crops, dim=0)

        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx:idx + self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        if self.dataset == "mot17":
            if self.test_dataset:
                path = self.reid_path_sbs_s50 or "external/weights/mot17_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = self.reid_path_sbs_s50 or "external/weights/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "dance":
            path = self.reid_path_sbs_s50 or "external/weights/dance_sbs_S50.pth"
        else:
            if self.reid_model_type == "sbs_s50" and self.reid_path_sbs_s50:
                path = self.reid_path_sbs_s50
            else:
                return self._get_general_model()

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def _get_general_model(self):
        if self.reid_model_type == "osnet":
            model = OSNetReID(embedding_dim=256)
            if self.reid_path_osnet:
                model.load_state_dict(torch.load(self.reid_path_osnet))
        elif self.reid_model_type == "sbs_s50":
            if not self.reid_path_sbs_s50:
                raise ValueError("SBS_S50 model path must be provided when reid_model_type is 'sbs_s50'")
            model = FastReID(self.reid_path_sbs_s50)
            model.half()
        elif self.reid_model_type == "ensemble":
            model_osnet = None
            model_sbs_s50 = None
            if self.reid_path_osnet:
                model_osnet = OSNetReID(embedding_dim=256)
                model_osnet.load_state_dict(torch.load(self.reid_path_osnet))
            if self.reid_path_sbs_s50:
                model_sbs_s50 = FastReID(self.reid_path_sbs_s50)
                model_sbs_s50.half()
            model = EnsembleOSNetReID(
                model_osnet,
                model_sbs_s50,
                self.reid_weight_osnet,
                self.reid_weight_sbs_s50,
                embedding_dim=256
            )
        else:
            raise ValueError("Invalid reid_model_type")

        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256) if self.reid_model_type in ["osnet", "ensemble"] else (128, 384)
        self.normalize = self.reid_model_type in ["osnet", "ensemble"]

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)