from mpi4py import MPI
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from PIL import Image
import re
import os
import ast
import time
import csv
import numpy as np
import pandas as pd
import hashlib
import datetime as date
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mpi4py import MPI
import sys
import random
import math
import hyperloglog 
from typing import List, Dict
import torch
import torch.nn as nn
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from transformers import AutoProcessor, AutoModel

os.makedirs('output', exist_ok=True)

# =========================================================
# MultimodalFusionRunner
# Algorithm 1: Multimodal Embedding and Fusion
# =========================================================
class MultimodalFusionRunner:
    def __init__(self):
        # ---------- PATHS ----------
        self.image_folder = "images"              # folder containing image files
        self.text_csv = "texts.csv"               # clinical CSV
        self.output_csv = "fused_embeddings.csv"  # final output
        self.output_dir = "offline_fused"

        self.batch_size = 16
        self.num_workers = 0   # IMPORTANT for Windows

        # ---------- MPI ----------
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()

        # ---------- DEVICE ----------
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            local_gpu = self.rank % num_gpus
            torch.cuda.set_device(local_gpu)
            self.device = torch.device(f"cuda:{local_gpu}")
            if self.rank == 0:
                print(f"Running on GPUs ({num_gpus} detected)")
        else:
            self.device = torch.device("cpu")
            if self.rank == 0:
                print("GPU not available — running on CPU")

        # ---------- SETUP ----------
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        self.comm.Barrier()

        # ---------- LOAD ----------
        self._load_dataset()
        self._load_models()

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    def _load_dataset(self):
        df = pd.read_csv(
            self.text_csv,
            encoding="latin1",
            engine="python"
        )

        required_cols = [
            "filename",      # ACTUAL image filename without extension
            "MeSH",
            "Problems",
            "image",         # image description / view
            "indication",
            "comparison",
            "findings",
            "impression"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ---------- IMAGE PATHS ----------
        self.image_paths = []
        for fname in df["filename"].astype(str).tolist():
            found = False
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(self.image_folder, fname + ext)
                if os.path.exists(candidate):
                    self.image_paths.append(candidate)
                    found = True
                    break

            if not found:
                raise FileNotFoundError(
                    f"No image file found for filename='{fname}' "
                    f"(expected {fname}.jpg/.png/.jpeg in {self.image_folder})"
                )

        # ---------- VERIFY IMAGE FILES ----------
        missing = [p for p in self.image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image files not found. Example: {missing[:3]}"
            )

        # ---------- BUILD CLINICAL TEXT ----------
        def build_text(row):
            parts = [
                f"MeSH: {row['MeSH']}" if pd.notna(row["MeSH"]) else "",
                f"Problems: {row['Problems']}" if pd.notna(row["Problems"]) else "",
                f"ImageDescription: {row['image']}" if pd.notna(row["image"]) else "",
                f"Indication: {row['indication']}" if pd.notna(row["indication"]) else "",
                f"Comparison: {row['comparison']}" if pd.notna(row["comparison"]) else "",
                f"Findings: {row['findings']}" if pd.notna(row["findings"]) else "",
                f"Impression: {row['impression']}" if pd.notna(row["impression"]) else "",
            ]
            return " ".join([p for p in parts if p.strip()])

        self.texts = df.apply(build_text, axis=1).tolist()

        if len(self.image_paths) != len(self.texts):
            raise ValueError("Mismatch between number of images and text entries.")

        if self.rank == 0:
            print(f"Loaded {len(self.image_paths)} samples using filename column")

    # -----------------------------------------------------
    # Load MedCLIP
    # -----------------------------------------------------
    def _load_models(self):
        # Same multimodal model for both image and text
        self.processor = AutoProcessor.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

        self.model = AutoModel.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

        self.model.eval().to(self.device)

        # Infer embedding dimension once
        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            dummy_text = ["test"]
            dummy_inputs = self.processor(
                text=dummy_text,
                images=[dummy_img],
                return_tensors="pt",
                padding=True
            )
            dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
            img_embeds, text_embeds = self.model(**dummy_inputs)
            self.embed_dim = img_embeds.shape[1]
            self.fused_dim = self.embed_dim * 2

        if self.rank == 0:
            print(f"MedCLIP loaded successfully. Embedding dim = {self.embed_dim}")

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    class _Dataset(Dataset):
        def __init__(self, image_paths, texts):
            self.image_paths = image_paths
            self.texts = texts

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            txt = self.texts[idx]
            return img, txt

    # -----------------------------------------------------
    # Custom collate function
    # -----------------------------------------------------
    @staticmethod
    def _collate_fn(batch):
        images, texts = zip(*batch)
        return list(images), list(texts)

    # -----------------------------------------------------
    # MPI sharding
    # -----------------------------------------------------
    def _shard_data(self):
        n = len(self.image_paths)
        shard_size = (n + self.world_size - 1) // self.world_size
        start = self.rank * shard_size
        end = min(start + shard_size, n)

        return self.image_paths[start:end], self.texts[start:end]

    # -----------------------------------------------------
    # Run MedCLIP fusion
    # -----------------------------------------------------
    def _run_fusion(self):
        local_imgs, local_txts = self._shard_data()

        dataset = self._Dataset(local_imgs, local_txts)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._collate_fn
        )

        fused_local = []

        with torch.no_grad():
            for images, texts in loader:
                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # MedCLIP gives aligned embeddings for both modalities
                v_img, v_txt = self.model(**inputs)   # both: (B, D)

                # Concatenate + normalize
                v_cat = torch.cat([v_img, v_txt], dim=1)    # (B, 2D)
                v_fuse = F.normalize(v_cat, p=2, dim=1)

                fused_local.append(v_fuse.cpu())

        if len(fused_local) == 0:
            fused_local = torch.empty((0, self.fused_dim), dtype=torch.float32)
        else:
            fused_local = torch.cat(fused_local, dim=0)

        torch.save(
            fused_local,
            os.path.join(self.output_dir, f"fused_rank{self.rank}.pt")
        )

        print(f"[Rank {self.rank}] Saved {fused_local.shape[0]} embeddings")
        return fused_local

    # -----------------------------------------------------
    # Merge and save CSV (rank 0)
    # -----------------------------------------------------
    def _merge_and_save_csv(self, gathered):
        if len(gathered) == 0:
            raise ValueError("No gathered embeddings found.")

        f_all = np.vstack(gathered)

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["vector_id"] + [f"dim_{i}" for i in range(f_all.shape[1])]
            writer.writerow(header)
            for i, row in enumerate(f_all):
                writer.writerow([i] + row.tolist())

        print(f"Combined CSV saved to: {self.output_csv}")
        print(f"Final shape: {f_all.shape}")

    # -----------------------------------------------------
    # Public entry point
    # -----------------------------------------------------
    def execute(self):
        self.comm.Barrier()
        start = time.time()

        f_local = self._run_fusion()

        self.comm.Barrier()
        end = time.time()

        local_time = end - start
        max_time = self.comm.reduce(local_time, op=MPI.MAX, root=0)

        gathered = self.comm.gather(f_local.numpy(), root=0)

        if self.rank == 0:
            print("\n=== OFFLINE CONVERSION COMPLETED ===")
            print(f"Wall-clock time (slowest rank): {max_time:.2f} seconds")
            self._merge_and_save_csv(gathered)

class FusedSplitRunner:
    """
    Splits fused embeddings into train/test CSV files
    """

    def __init__(
        self,
        fused_csv="fused_embeddings.csv",
        train_csv="fused_train.csv",
        test_csv="fused_test.csv",
        test_ratio=0.5,
        random_state=42
    ):
        self.fused_csv = fused_csv
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.test_ratio = test_ratio
        self.random_state = random_state

    def execute(self):
        # Load fused embeddings
        df = pd.read_csv(self.fused_csv)

        if len(df) == 0:
            raise ValueError("Fused CSV is empty")

        # Split
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            random_state=self.random_state,
            shuffle=True
        )

        # Save
        train_df.to_csv(self.train_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

        print("\n=== FUSED DATASET SPLIT COMPLETED ===")
        print(f"Input file : {self.fused_csv}")
        print(f"Train file : {self.train_csv} | shape = {train_df.shape}")
        print(f"Test file  : {self.test_csv}  | shape = {test_df.shape}")

class Preprocessor:
    def __init__(self, filepath, n_components=180, mse_threshold=0.01):
        self.filepath = filepath
        self.n_components = n_components
        self.mse_threshold = mse_threshold

    def load_and_scale(self):
        df = pd.read_csv(self.filepath)
        #features = df.drop(['Activity'], axis=1, errors='ignore')
        features = df.drop(['vector_id'], axis=1, errors='ignore')
        return StandardScaler().fit_transform(features)

    def apply_pca_and_check(self, data):
        pca = PCA(n_components=self.n_components)
        reduced = pca.fit_transform(data)
        mse = mean_squared_error(data, pca.inverse_transform(reduced))
        return mse < self.mse_threshold, reduced, mse

class Network:
    def __init__(self, num_nodes_per_cluster=5, total_clusters=1):
        self.num_nodes_per_cluster = num_nodes_per_cluster
        self.total_clusters = total_clusters
        self.clusters = self._create_clusters()

    def _create_clusters(self):
        clusters = {}
        node_id = 0
        for cluster_id in range(self.total_clusters):
            clusters[cluster_id] = []
            for _ in range(self.num_nodes_per_cluster):
                clusters[cluster_id].append(node_id)
                node_id += 1
        return clusters

    def get_nodes_for_cluster(self, cluster_id):
        return self.clusters.get(cluster_id, [])

    def display(self):
        pass

class ExpanderOverlay:
    """
    Random d-regular overlay (expander-like with high probability)
    """
    def __init__(self, n: int, d: int = 4, seed: int = 42, max_tries: int = 50):
        if d <= 0:
            raise ValueError("d must be > 0")
        if d >= n:
            raise ValueError("d must be < n")
        if (n * d) % 2 != 0:
            raise ValueError("n*d must be even")

        self.n = n
        self.d = d
        self.seed = seed
        self.neighbors: List[List[int]] = self._build(max_tries)

    def _build(self, max_tries: int) -> List[List[int]]:
        for attempt in range(max_tries):
            rng = random.Random(self.seed + attempt)
            stubs = []
            for i in range(self.n):
                stubs.extend([i] * self.d)
            rng.shuffle(stubs)

            adj = [set() for _ in range(self.n)]
            ok = True

            for k in range(0, len(stubs), 2):
                a, b = stubs[k], stubs[k + 1]
                if (
                    a == b
                    or b in adj[a]
                    or len(adj[a]) >= self.d
                    or len(adj[b]) >= self.d
                ):
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)

            if ok and all(len(adj[i]) == self.d for i in range(self.n)):
                return [sorted(list(s)) for s in adj]

        raise RuntimeError("Failed to build expander overlay")

    def rounds(self) -> int:
        return int(math.ceil(math.log2(self.n))) + 1

def expander_gossip_maps(
    initial_maps: List[Dict],
    neighbors: List[List[int]],
    rounds: int
) -> List[Dict]:
    """
    Expander gossip:
    Each node i starts with initial_maps[i] (a dict),
    and for 'rounds' iterations unions knowledge from neighbors.
    """
    known = [dict(m) for m in initial_maps]
    n = len(known)

    for _ in range(rounds):
        new_known = [dict(k) for k in known]
        for i in range(n):
            for nb in neighbors[i]:
                new_known[i].update(known[nb])
        known = new_known

    return known

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        return hashlib.sha256((str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)).encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, date.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_valid(self):
        for i in range(1, len(self.chain)):
            if self.chain[i].hash != self.chain[i].calculate_hash():
                return False
            if self.chain[i].previous_hash != self.chain[i - 1].hash:
                return False
        return True

    def get_chain(self):
        with open('output/Blockchain.txt', 'w') as f:
            for block in self.chain:
                f.write(f"Block #{block.index}\nTimestamp: {block.timestamp}\nHash: {block.hash}\nPrevious Hash: {block.previous_hash}\nData: {block.data}\n\n")

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        num_faulty_nodes = int(sub_cluster_size * fault_percentage)
        faulty_nodes = set(random.sample(range(sub_cluster_size), num_faulty_nodes))
        print(f"Fault percentage: {fault_percentage * 100:.2f}%")
        print(f"Number of faulty nodes: {num_faulty_nodes} out of {sub_cluster_size}")
        return faulty_nodes

    def consensus_success_rate(self, n, f, t):
        h = n - f
        failure_probability = 0
        if f >= t:
            return 0.0
        for x in range(t, f + 1):
            failure_probability += math.comb(f, x) * (0.5 ** x) * (0.5 ** (f - x))
        success_rate = 1 - failure_probability
        return success_rate

    def consensus(self, block, rank, fault_percentage, sub_cluster_size=10):
        parsed_data = json.loads(block.data)
        data_hash = hash_data(parsed_data['data'])
        blk_hash = parsed_data['hash']

        hll_ok = True
        if 'hll_estimate' in parsed_data:
          est_commit = float(parsed_data['hll_estimate'])
          # Recompute with pip HLL
          hll_check = hyperloglog.HyperLogLog(parsed_data.get('error_rate', 0.01))
          for item in parsed_data['data']:
             hll_check.add(data_to_bytes(item))
          est = len(hll_check)
        # relative tolerance ~ error_rate * 3 sigma
          rel_tol = 3.0 * parsed_data.get('error_rate', 0.01)
          hll_ok = abs(est - est_commit) <= rel_tol * max(est_commit, 1.0)

        expander_degree = 4
        overlay = ExpanderOverlay(sub_cluster_size, d=expander_degree, seed=123)
        R = overlay.rounds()

        votes = np.zeros(sub_cluster_size, dtype=bool)
        temp_commit_data = [None] * sub_cluster_size

        faulty_nodes = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)

        # ------------------ Push Phase (Prepare + Vote Gossip) ------------------
        push_start = time.time()

        def local_prepare(i):
            if i in faulty_nodes:
                votes[i] = False
                temp_commit_data[i] = None
                return

            if (data_hash == blk_hash) and hll_ok:
                temp = parsed_data.copy()
                temp['meta'] = {
                    'validator_node': i,
                    'coordinator_rank': rank,
                    'prepare_time': str(date.datetime.now()),
                    'status': 'PREPARED'
                }
                temp_commit_data[i] = temp
                votes[i] = True
            else:
                votes[i] = False
                temp_commit_data[i] = None

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(local_prepare, range(sub_cluster_size))
        initial = [{i: bool(votes[i])} for i in range(sub_cluster_size)]
        known_votes = expander_gossip_maps(initial, overlay.neighbors, rounds=R)
        
        coordinator_node = 0
        coord_view = known_votes[coordinator_node]
        yes_votes_seen = sum(1 for v in coord_view.values() if v)

        committed = False
        if yes_votes_seen >= int(sub_cluster_size * 0.51):
            coordinator_block = parsed_data.copy()
            coordinator_block['meta'] = {
                'coordinator_rank': rank,
                'coordinator_node': coordinator_node,
                'commit_time': str(date.datetime.now()),
                'status': 'COORDINATOR_COMMITTED',
                'votes_seen': yes_votes_seen,
                'overlay_degree': expander_degree,
                'gossip_rounds': R
            }
            out_path = f'output/coordinator_commit_rank_{rank}.csv'
            df = pd.DataFrame([coordinator_block])

            header = not os.path.exists(out_path)
            df.to_csv(out_path, mode="a", header=header, index=False)
            committed = True

        push_end = time.time()
        push_duration = push_end - push_start

        # ------------------ Pull Phase (Commit Decision Gossip) ------------------
        pull_start = time.time()

        # Disseminate the final decision over the expander (each node starts knowing nothing except leader)
        # We gossip a dict {"DECISION": True/False} from coordinator_node outward.
        decision_key = "DECISION"
        decision_initial = [{} for _ in range(sub_cluster_size)]
        decision_initial[coordinator_node] = {decision_key: committed}
        decision_known = expander_gossip_maps(decision_initial, overlay.neighbors, rounds=R)

        def apply_commit(i):
            # Each node checks if it learned the decision.
            decided = decision_known[i].get(decision_key, False)

            if votes[i]:
                # If prepared and decision is commit, finalize commit metadata
                if decided and temp_commit_data[i] is not None:
                    temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                    temp_commit_data[i]['meta']['status'] = 'COMMITTED'
                elif temp_commit_data[i] is not None:
                    temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                    temp_commit_data[i]['meta']['status'] = 'ABORTED'
            else:
                # Nodes that did not prepare (or were faulty) can still learn commit and update from ledger
                if decided:
                    fallback = parsed_data.copy()
                    fallback['meta'] = {
                        'coordinator_rank': rank,
                        'node_id': i,
                        'commit_time': str(date.datetime.now()),
                        'status': 'COMMIT_READ_FROM_LEDGER'
                    }
                    temp_commit_data[i] = fallback
                else:
                    # No commit happened; keep a record if you want
                    fallback = parsed_data.copy()
                    fallback['meta'] = {
                        'coordinator_rank': rank,
                        'node_id': i,
                        'commit_time': str(date.datetime.now()),
                        'status': 'REJECTED'
                    }
                    temp_commit_data[i] = fallback

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(apply_commit, range(sub_cluster_size))

        pull_end = time.time()
        pull_duration = pull_end - pull_start

        pd.DataFrame(temp_commit_data).to_csv(f'output/subcluster_all_nodes_coordinator_{rank}.csv', index=False)

        if committed:
            self.add_block(block)

        consensus_rate = self.consensus_success_rate(sub_cluster_size, len(faulty_nodes), int(sub_cluster_size * 0.51))
        print(f"Consensus Success Rate: {consensus_rate * 100:.2f}%")
        print(f"[Expander] degree={expander_degree}, rounds={R}, coordinator_node={coordinator_node}, yes_seen={yes_votes_seen}")

        return committed, push_duration, pull_duration



def hash_data(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

def data_to_bytes(item, float_ndigits=6):
    # Normalize floats for stability
    def norm(x):
        if isinstance(x, float):
            return round(x, float_ndigits)
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    canon = json.dumps(norm(item), separators=(',', ':'), ensure_ascii=False)
    return hashlib.blake2b(canon.encode('utf-8'), digest_size=8).digest()

def data_validation(block):
    try:
        content = json.loads(block.data)
        return hash_data(content['data']) == content['hash']
    except:
        return False
    
class KMeansProcessor:
    def __init__(self, k=3, num_steps=100, seed=42):
        self.k = k
        self.num_steps = num_steps
        self.seed = seed

    def initialize_centroids(self, data):
        n = data.shape[0]
        if n < self.k:
           raise ValueError(
              f"KMeans init failed: k={self.k} but only {n} samples available"
        )
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=self.k, replace=False)
        return data[indices].astype(np.float64)


    def assign_clusters(self, data, centroids):
        # distances: (n_local, k)
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def run(self, local_data, comm, global_data=None):
        """
        Correct distributed K-means:
        - rank0 initializes centroids from global_data
        - each iter:
            local assigns -> local sums + local counts
            Allreduce sums + Allreduce counts
            centroids = global_sums / global_counts  (per cluster)
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Ensure float64 for stable centroid updates
        local_data = np.asarray(local_data, dtype=np.float64)

        # Init centroids on rank 0 from full dataset
        if rank == 0:
            if global_data is None:
                raise ValueError("global_data must be provided on rank 0 for centroid init.")
            centroids = self.initialize_centroids(np.asarray(global_data, dtype=np.float64))
        else:
            centroids = None

        centroids = comm.bcast(centroids, root=0)  # shape (k, d)
        k, d = centroids.shape

        for _ in range(self.num_steps):
            # Assign
            labels = self.assign_clusters(local_data, centroids)

            # Local sums and counts
            local_sums = np.zeros((k, d), dtype=np.float64)
            local_counts = np.zeros(k, dtype=np.int64)

            for j in range(k):
                mask = (labels == j)
                cnt = int(mask.sum())
                local_counts[j] = cnt
                if cnt > 0:
                    local_sums[j] = local_data[mask].sum(axis=0)

            # Global sums and counts
            global_sums = np.zeros_like(local_sums)
            global_counts = np.zeros_like(local_counts)

            comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
            comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

            # Update centroids safely (avoid division by 0)
            new_centroids = centroids.copy()
            nonempty = global_counts > 0
            new_centroids[nonempty] = global_sums[nonempty] / global_counts[nonempty][:, None]

            # Optional: handle empty clusters by re-seeding from rank0/global_data
            # (simple safe fallback)
            if rank == 0 and np.any(~nonempty):
                empty_idx = np.where(~nonempty)[0]
                # re-pick random points from global_data for those empty clusters
                rng = np.random.default_rng(self.seed)
                repl = rng.choice(np.asarray(global_data).shape[0], size=len(empty_idx), replace=False)
                new_centroids[empty_idx] = np.asarray(global_data, dtype=np.float64)[repl]

            # Broadcast the repaired centroids (in case rank0 reseeded)
            new_centroids = comm.bcast(new_centroids, root=0)
            centroids = new_centroids

        # Final labels returned with local data
        local_counts = np.bincount(labels, minlength=self.k).astype(np.int64)
        global_counts = np.zeros_like(local_counts)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

        clustered_data = np.column_stack((local_data, labels))
        return clustered_data, centroids, global_counts, local_counts

class KMeansRunner:
    def __init__(self, filepath, k=3, num_steps=100):
        self.filepath = filepath
        self.processor = KMeansProcessor(k, num_steps)

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = MPI.Wtime()

        pre = Preprocessor(self.filepath)
        data = pre.load_and_scale()
        fault_percentage = random.random() * 0.49

        if rank == 0:
            #passed, reduced, mse = pre.apply_pca_and_check(data)
            train_data = data
            #if not passed:
                #print(f"PCA MSE too high: {mse:.6f}")
                #sys.exit()
        else:
            train_data = None
            # reduced = None

        fault_percentage = comm.bcast(fault_percentage, root=0)
        train_data =  comm.bcast(train_data, root=0)
        #reduced = comm.bcast(reduced, root=0)
        #local_data = ColumnShardProcessor.distribute_columns(reduced, comm)
        #local_data = np.array_split(reduced, comm.Get_size())[rank]
        #clustered_data = self.processor.run(local_data, comm, reduced)
        
        local_data = np.array_split(train_data, comm.Get_size())[rank]
        clustered_data, centroids, global_counts, local_counts = self.processor.run(local_data, comm, train_data)
        # Rank 0 only (authoritative)
        if rank == 0:
          np.save("output/centroids.npy", centroids)
          np.save("output/shard_counts.npy", global_counts)

        # Every rank (optional diagnostics)
        np.save(f"output/local_cluster_counts_rank{rank}.npy", local_counts)

        blockchain = Blockchain()

        # HLL: build sketch for this block's data
        hll = hyperloglog.HyperLogLog(0.01)   # 1% error rate
        for item in clustered_data.tolist():
          hll.add(data_to_bytes(item)) 


        blk_data = {
            'coordinator': rank,
            'data': clustered_data.tolist(),
            'hash': hash_data(clustered_data.tolist()),
            'hll_estimate': len(hll),   
            'error_rate': 0.01                  
        }

        blk = Block(rank + 1, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)

        end = MPI.Wtime()

        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)

        all_blocks = comm.gather(blk if committed else None, root=0)

        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            #total_vectors = reduced.shape[0]
            total_vectors = train_data.shape[0]
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)

            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[K-means mode] Execution Time: {end - start:.4f} sec")
            print(f"[K-means mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")

class ColumnShardProcessor:
    @staticmethod
    def distribute_columns(data, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_cols = data.shape[1]
        per = n_cols // size
        rem = n_cols % size
        start = rank * per + min(rank, rem)
        end = start + per + (1 if rank < rem else 0)
        return data[:, start:end]
    
class ColumnShardRunner:
    def __init__(self, filepath):
        self.filepath = filepath

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = MPI.Wtime()

        pre = Preprocessor(self.filepath)
        data = pre.load_and_scale()
        fault_percentage = random.random() * 0.49

        if rank == 0:
            passed, reduced, mse = pre.apply_pca_and_check(data)
            if not passed:
                print(f"PCA MSE too high: {mse:.6f}")
                sys.exit()
        else:
            reduced = None
        fault_percentage = comm.bcast(fault_percentage, root=0)
        reduced = comm.bcast(reduced, root=0)
        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)
        blockchain = Blockchain()

        hll = hyperloglog.HyperLogLog(0.01)
        for item in local_data.tolist():
          hll.add(data_to_bytes(item))

        blk_data = {
            'coordinator': rank,
            'data': local_data.tolist(),
            'hash': hash_data(local_data.tolist()),
            'hll_estimate': len(hll),
            'error_rate': 0.01
        }

        blk = Block(rank + 1001, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)

        end = MPI.Wtime()

        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)

        all_blocks = comm.gather(blk if committed else None, root=0)
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            total_vectors = reduced.shape[0]
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)

            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[Column mode] Execution Time: {end - start:.4f} sec")
            print(f"[Column mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")

class MPIOnlineRouterTesterHuge:

    def __init__(
        self,
        base_dir="output",
        fused_csv="fused_test.csv",
        mlp_path="mlp_router.pt",
        centroids_path="centroids.npy",
        counts_path="shard_counts.npy",
        aging_T=1000,
        aging_beta=0.01,
        chunk_rows=50000,
        device="cpu",
    ):
        # ---------- MPI ----------
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world = self.comm.Get_size()

        # ---------- PATHS ----------
        self.base_dir = base_dir
        self.fused_csv = os.path.join(base_dir, fused_csv)
        self.mlp_path = os.path.join(base_dir, mlp_path)
        self.centroids_path = os.path.join(base_dir, centroids_path)
        self.counts_path = os.path.join(base_dir, counts_path)

        # ---------- PARAMS ----------
        self.T = float(aging_T)
        self.beta = float(aging_beta)
        self.chunk_rows = int(chunk_rows)
        self.device = device

        # ---------- LOAD CENTROIDS ----------
        if self.rank == 0:
            self.C = np.load(self.centroids_path).astype(np.float32)
            self.N = np.load(self.counts_path).astype(np.int64)
            self.k, self.d = self.C.shape
        else:
            self.C = None
            self.N = None
            self.k = None
            self.d = None

        self.k = self.comm.bcast(self.k, root=0)
        self.d = self.comm.bcast(self.d, root=0)

        if self.rank != 0:
            self.C = np.empty((self.k, self.d), dtype=np.float32)
            self.N = np.empty((self.k,), dtype=np.int64)

        self.comm.Bcast(self.C, root=0)
        self.comm.Bcast(self.N, root=0)

        # ---------- LOAD MLP ----------
        self.mlp = self._load_mlp_router(self.mlp_path).to(self.device)
        self.mlp.eval()

        if self.rank == 0:
            print(f"[OK] Router loaded | k={self.k}, d={self.d}")
            print("[MODE] Router-only (centroids frozen)")

    # =====================================================
    def _load_mlp_router(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        input_dim = ckpt["input_dim"]
        num_classes = ckpt["num_classes"]

        if input_dim != self.d:
            raise ValueError("MLP input_dim mismatch")

        class MLPRouter(torch.nn.Module):
            def __init__(self, d, k):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(d, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, k),
                )

            def forward(self, x):
                return self.net(x)

        model = MLPRouter(input_dim, num_classes)
        model.load_state_dict(ckpt["state_dict"])
        return model

    # =====================================================
    def _aging_weights(self):
        return 1.0 + np.exp(self.beta * (self.N.astype(np.float32) - self.T))

    # =====================================================
    def _assign_batch(self, X):
        W = self._aging_weights()
        assigned = np.empty((X.shape[0],), dtype=np.int64)
        mlp_labels = np.empty((X.shape[0],), dtype=np.int64)

        with torch.no_grad():
            logits = self.mlp(torch.from_numpy(X).to(self.device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            mlp_labels[:] = np.argmax(probs, axis=1)

            for i, v in enumerate(X):
                diffs = self.C - v
                dists = np.sqrt((diffs * diffs).sum(axis=1))
                scores = dists * W
                assigned[i] = int(np.argmin(scores))

        return assigned, mlp_labels

    # =====================================================
    def execute(self):
        if self.rank == 0:
            reader = pd.read_csv(self.fused_csv, chunksize=self.chunk_rows)
        else:
            reader = None

        while True:
            if self.rank == 0:
                try:
                    chunk = next(reader)

                    if "vector_id" not in chunk.columns:
                        chunk.insert(0, "vector_id", np.arange(len(chunk)))

                    dim_cols = sorted(
                        [c for c in chunk.columns if c.startswith("dim_")],
                        key=lambda x: int(x.split("_")[1])
                    )

                    if len(dim_cols) != self.d:
                        raise ValueError("Dimension mismatch")

                    ids = chunk["vector_id"].to_numpy(np.int64)
                    X = chunk[dim_cols].to_numpy(np.float32)

                except StopIteration:
                    ids = None
                    X = None
            else:
                ids = None
                X = None

            done = self.comm.bcast(1 if ids is None else 0, root=0)
            if done:
                break

            ids = self.comm.bcast(ids, root=0)
            X = self.comm.bcast(X, root=0)

            assigned, mlp_labels = self._assign_batch(X)

            if self.rank == 0:
                out_df = pd.DataFrame({
                    "vector_id": ids,
                    "predicted_label": mlp_labels,
                    "assigned_shard": assigned
                })

                out_path = os.path.join(self.base_dir, "router_predictions.csv")
                header = not os.path.exists(out_path)
                out_df.to_csv(out_path, mode="a", header=header, index=False)

                print(f"[Chunk] {len(X)} vectors routed")

        if self.rank == 0:
            print("[DONE] Router testing completed (no centroid updates)")


class ShardBuilder:
    def __init__(
        self,
        input_dir: str,
        output_file: str = "final_shard_1280.csv",
        coordinator_column: str = "coordinator",
        data_column: str = "data",
        target_dim: int = 1280
    ):
        self.input_dir = input_dir
        self.output_file = output_file
        self.coordinator_column = coordinator_column
        self.data_column = data_column
        self.target_dim = target_dim
        self.file_pattern = re.compile(r"coordinator_commit_rank_\d+\.csv")

    # -------------------------
    def _fix_dim(self, vec):
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if len(vec) > self.target_dim:
            return vec[:self.target_dim]
        if len(vec) < self.target_dim:
            return np.pad(vec, (0, self.target_dim - len(vec)))
        return vec

    # -------------------------
    def _parse_vectors(self, cell):
        """
        Returns a LIST of vectors
        """
        if not isinstance(cell, str):
            return []

        parsed = ast.literal_eval(cell)

        # Case: [[v1], [v2], ...]
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            return parsed

        # Case: [v]
        if isinstance(parsed, list):
            return [parsed]

        return []

    # -------------------------
    def execute(self):
        all_features = []
        all_labels = []

        files = sorted(
            f for f in os.listdir(self.input_dir)
            if self.file_pattern.fullmatch(f)
        )

        if not files:
            raise RuntimeError("No coordinator_commit_rank_*.csv files found")

        for fname in files:
            path = os.path.join(self.input_dir, fname)
            print(f"[ShardBuilder] Processing {path}")

            df = pd.read_csv(path)

            for _, row in df.iterrows():
                label = row[self.coordinator_column]
                vectors = self._parse_vectors(row[self.data_column])

                for vec in vectors:
                    vec = self._fix_dim(vec)
                    all_features.append(vec)
                    all_labels.append(label)

        X = np.asarray(all_features, dtype=np.float32)
        y = np.asarray(all_labels)

        final_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(self.target_dim)])
        final_df.insert(0, "label", y)

        final_df.to_csv(self.output_file, index=False)

        print("===================================")
        print("DONE")
        print("Output:", self.output_file)
        print("Shape :", final_df.shape)
        print("===================================")

class OnlineInsertion:

    def __init__(
        self,
        fused_csv="output/fused_test.csv",
        routing_csv="output/router_predictions.csv",
        fault_percentage=0.2,
        sub_cluster_size=10,
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world = self.comm.Get_size()

        self.fused_csv = fused_csv
        self.routing_csv = routing_csv
        self.fault_percentage = fault_percentage
        self.sub_cluster_size = sub_cluster_size

        self.blockchain = Blockchain()

    def execute(self):
        # ---------------- Rank 0 loads routing ----------------
        if self.rank == 0:
            fused = pd.read_csv(self.fused_csv)
            routing = pd.read_csv(self.routing_csv)
        else:
            fused = None
            routing = None

        fused = self.comm.bcast(fused, root=0)
        routing = self.comm.bcast(routing, root=0)

        # ---------------- Select vectors for THIS shard ----------------
        local_rows = routing[routing["assigned_shard"] == self.rank]

        if len(local_rows) == 0:
            print(f"[Rank {self.rank}] No vectors assigned")
            return

        vector_ids = local_rows["vector_id"].values

        dim_cols = [c for c in fused.columns if c.startswith("dim_")]
        mask = fused["vector_id"].isin(vector_ids)
        X_local = fused.loc[mask, dim_cols].to_numpy(dtype=np.float32)


        # ---------------- Build blockchain block ----------------
        blk_data = {
            "coordinator": self.rank,
            "data": X_local.tolist(),
            "hash": hash_data(X_local.tolist()),
        }

        block = Block(
            index=len(self.blockchain.chain),
            timestamp=date.datetime.now(),
            data=json.dumps(blk_data),
            previous_hash=self.blockchain.get_latest_block().hash
        )

        # ---------------- Consensus ----------------
        committed, push_t, pull_t = self.blockchain.consensus(
            block,
            rank=self.rank,
            fault_percentage=self.fault_percentage,
            sub_cluster_size=self.sub_cluster_size
        )

        # ---------------- Result ----------------
        if committed:
            print(
                f"[Rank {self.rank}] COMMITTED {len(X_local)} vectors "
                f"(push={push_t:.4f}s, pull={pull_t:.4f}s)"
            )
        else:
            print(f"[Rank {self.rank}] ABORTED insertion")


# =========================================================
# MAIN (ONE LINE EXECUTION)
# =========================================================
if __name__ == "__main__":
    filepath = 'fused_embeddings.csv'
    MultimodalFusionRunner().execute()
    #KMeansRunner(filepath).execute()
    #MLPRouterTrainerMPI("final_shard_1280.csv").execute()
    #ShardBuilder("output").execute()
    #MPIOnlineRouterTesterHuge(
    #base_dir="output",
    #fused_csv="fused_test.csv",
    #mlp_path="mlp_router.pt",
    #centroids_path="centroids.npy",
    #counts_path="shard_counts.npy"
#).execute()
    #OnlineInsertion(
    #fused_csv="output/fused_test.csv",
    #routing_csv="output/router_predictions.csv",
    #fault_percentage=0.25,
    #sub_cluster_size=10
#).execute()
