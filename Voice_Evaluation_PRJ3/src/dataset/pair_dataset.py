import random
from torch.utils.data import Dataset
from src.dataset.voice_dataset import VoiceRankingDataset


class VoicePairDataset(Dataset):
    def __init__(self, csv_path, audio_dir, num_pairs=50000,
                 hard_gap=100, easy_gap=500):
        self.base_ds = VoiceRankingDataset(csv_path, audio_dir)
        self.num_pairs = num_pairs
        self.hard_gap = hard_gap
        self.easy_gap = easy_gap

        self.indices = list(range(len(self.base_ds)))
        # IMPORTANT: read ranks from metadata ONLY
        self.ranks = [int(row["rank"]) for row in self.base_ds.rows]

    def __len__(self):
        return self.num_pairs

    def _sample_pair(self, hard=True, max_tries=50):
        for _ in range(max_tries):
            i, j = random.sample(self.indices, 2)
            gap = abs(self.ranks[i] - self.ranks[j])

            if hard and gap < self.hard_gap:
                return i, j
            if not hard and gap > self.easy_gap:
                return i, j

        return random.sample(self.indices, 2)

    def __getitem__(self, idx):
        hard = (idx % 2 == 0)
        i, j = self._sample_pair(hard=hard)

        si = self.base_ds[i]
        sj = self.base_ds[j]

        if si["rank"] < sj["rank"]:
            return {
                "features_a": si["features"],
                "features_b": sj["features"],
                "label": 1
            }
        else:
            return {
                "features_a": sj["features"],
                "features_b": si["features"],
                "label": 1
            }
