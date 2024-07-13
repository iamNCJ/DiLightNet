import json
import random
import cv2
import imageio
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


imageio.plugins.freeimage.download()


class RelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model='stabilityai/stable-diffusion-2-1',
            channel_aug_ratio=0.0,
            pred_normal_ratio=0.0,
            empty_prompt_ratio=0.0,
            self_ref_ratio=0.0,
            log_encode_hint=False,
            load_mask=False,
            use_black_image_filter=False,
            eval_mode=False,
            direct_data=None,
    ):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.channel_aug_ratio = channel_aug_ratio
        self.pred_normal_ratio = pred_normal_ratio
        self.empty_prompt_ratio = empty_prompt_ratio
        self.log_encode_hint = log_encode_hint
        self.load_mask = load_mask
        self.use_black_image_filter = use_black_image_filter
        self.eval_mode = eval_mode
        self.self_ref_ratio = self_ref_ratio
        if direct_data is not None:
            self.data = direct_data
        elif data_jsonl is not None:
            with open(data_jsonl, 'rt') as f:
                for line in tqdm(f, desc='Loading data'):
                    self.data.append(json.loads(line))
        else:
            raise ValueError("No data source specified.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            model_id = item['image'].split('/')[-4]

            target_filename = item['image']
            source_filename = item['ref'][0] if self.eval_mode else random.choice(item['ref'])
            if not self.eval_mode and random.random() < self.self_ref_ratio:
                source_filename = target_filename
            shading_filenames = item['hint']
            prompt = item['text']

            source = cv2.imread(source_filename)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = source / 255.  # Scale source images to [0, 1].

            target = cv2.imread(target_filename)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = target / 127.5 - 1.  # Normalize target images to [-1, 1].

            shadings = []
            use_pred_normal = random.random() < self.pred_normal_ratio
            for shading_filename in shading_filenames:
                if use_pred_normal:
                    shading_filename = shading_filename.replace('.png', '_gt_normal_smooth.png')
                    # shading_filename = shading_filename.replace('.png', '_pred_normal_smooth.png')
                hint_format = shading_filename.split('.')[-1]
                if hint_format == 'exr':
                    shading = imageio.v3.imread(shading_filename)
                else:
                    shading = cv2.imread(shading_filename)
                    shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)
                    shading = shading / 255.  # Scale shading images to [0, 1].
                shadings.append(shading)

            p = random.random()
            if p < self.channel_aug_ratio:
                channel_perm = np.random.permutation(3)
                source = source[..., channel_perm]
                target = target[..., channel_perm]
                for i in range(len(shadings)):
                    shadings[i] = shadings[i][..., channel_perm]
                prompt = ""  # remove prompt in case channel augmentation leads to different color
            shadings = np.concatenate(shadings, axis=2)
            shadings = shadings.astype(np.float32)
            if self.log_encode_hint:
                shadings = np.log(shadings + 1.)
            target = target.astype(np.float32)
            source = source.astype(np.float32)
            if self.use_black_image_filter:
                assert source.max(axis=2).mean() > 0.02, f"black ref image: {source_filename}"

            hint = np.concatenate([source, shadings], axis=2)

            depth_path = '/'.join(item['image'].split('/')[:-2] + ['depth0001.exr'])
            depth = imageio.v3.imread(depth_path)
            depth = depth[..., 0]
            mask = depth < 1e9
            if self.use_black_image_filter:
                assert np.mean(mask) > 0.1, f"low fg ratio: {source_filename}"
            if self.load_mask:
                hint = np.concatenate([mask.astype(np.float32)[..., None], hint], axis=2)

            # drop prompt in dataloader
            p = random.random()
            if p < self.empty_prompt_ratio:
                prompt = ""
            # tokenize prompt
            inputs = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            if self.eval_mode:
                raise e
            print(idx, repr(e), e)
            return self.__getitem__(np.random.randint(0, len(self.data)))

        return dict(
            pixel_values=target.transpose(2, 0, 1),
            input_ids=inputs.input_ids[0],
            conditioning_pixel_values=hint.transpose(2, 0, 1),
            text=prompt,
            target_file=target_filename,
            ref_file=source_filename,
            model_id=model_id,
            mask=mask
        )
