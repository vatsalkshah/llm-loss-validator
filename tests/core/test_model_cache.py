import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from src.core.model_cache import (
    ModelCacheManager,
    CachedModelInfo,
    EvictionStrategy,
)


class TestCachedModelInfo(unittest.TestCase):
    def test_dataclass_creation(self):
        info = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=time.time(),
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/cache",
            is_active=True,
            download_duration=10.5,
        )
        self.assertEqual(info.model_id, "test/model")
        self.assertEqual(info.model_type, "base")
        self.assertEqual(info.size_bytes, 1024)
        self.assertTrue(info.is_active)
        self.assertEqual(info.download_duration, 10.5)


class TestModelCacheManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.manifest_file = self.cache_dir / "test_manifest.json"

    def tearDown(self):
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization_default(self):
        with patch.dict(os.environ, {
            "CACHE_ENABLED": "true",
            "CACHE_DIR": str(self.cache_dir),
            "CACHE_MAX_SIZE_GB": "50",
            "CACHE_EVICTION_STRATEGY": "LRU",
        }):
            manager = ModelCacheManager(manifest_file=str(self.manifest_file))
            
            self.assertTrue(manager.enabled)
            self.assertEqual(manager.cache_dir, self.cache_dir)
            self.assertEqual(manager.max_size_bytes, 50 * 1024 ** 3)
            self.assertEqual(manager.eviction_strategy, EvictionStrategy.LRU)
            self.assertTrue(self.cache_dir.exists())

    def test_initialization_disabled(self):
        with patch.dict(os.environ, {"CACHE_ENABLED": "false"}):
            manager = ModelCacheManager(
                cache_dir=str(self.cache_dir),
                manifest_file=str(self.manifest_file),
            )
            self.assertFalse(manager.enabled)

    def test_make_cache_key(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        key = manager._make_cache_key("meta-llama/Llama-2-7b", "main")
        self.assertEqual(key, "meta-llama/Llama-2-7b@main")

    def test_parse_cache_key(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        model_id, revision = manager._parse_cache_key("meta-llama/Llama-2-7b@v1.0")
        self.assertEqual(model_id, "meta-llama/Llama-2-7b")
        self.assertEqual(revision, "v1.0")

    def test_save_and_load_manifest(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        cache_info = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=2048,
            download_timestamp=5.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/test",
            is_active=False,
        )
        
        manager.manifest["test/model@main"] = cache_info
        manager._save_manifest()
        
        self.assertTrue(self.manifest_file.exists())
        
        new_manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        self.assertEqual(len(new_manager.manifest), 1)
        loaded_info = new_manager.manifest["test/model@main"]
        self.assertEqual(loaded_info.model_id, "test/model")
        self.assertEqual(loaded_info.size_bytes, 2048)

    def test_get_cache_size(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["model1@main"] = CachedModelInfo(
            model_id="model1",
            model_type="base",
            size_bytes=1000,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/m1",
        )
        
        manager.manifest["model2@main"] = CachedModelInfo(
            model_id="model2",
            model_type="lora",
            size_bytes=500,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/m2",
        )
        
        self.assertEqual(manager.get_cache_size(), 1500)

    def test_get_cache_stats(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=10,
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["model1@main"] = CachedModelInfo(
            model_id="model1",
            model_type="base",
            size_bytes=1024 ** 3,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/m1",
            is_active=True,
        )
        
        stats = manager.get_cache_stats()
        
        self.assertEqual(stats["model_count"], 1)
        self.assertEqual(stats["active_count"], 1)
        self.assertAlmostEqual(stats["total_size_gb"], 1.0, places=2)
        self.assertEqual(stats["max_size_gb"], 10)

    def test_is_model_cached(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/test",
        )
        
        self.assertTrue(manager.is_model_cached("test/model", "main"))
        self.assertFalse(manager.is_model_cached("test/model", "v1.0"))
        self.assertFalse(manager.is_model_cached("other/model", "main"))

    def test_mark_active(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        initial_time = time.time()
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=initial_time,
            revision="main",
            cache_path="/tmp/test",
            is_active=False,
        )
        
        time.sleep(0.1)
        result = manager.mark_active("test/model", "main", active=True)
        
        self.assertTrue(result)
        self.assertTrue(manager.manifest["test/model@main"].is_active)
        self.assertGreater(
            manager.manifest["test/model@main"].last_accessed,
            initial_time
        )

    def test_mark_active_unknown_model(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        result = manager.mark_active("unknown/model", "main")
        self.assertFalse(result)

    def test_list_models(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["model1@main"] = CachedModelInfo(
            model_id="model1",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/m1",
            is_active=True,
        )
        
        manager.manifest["model2@main"] = CachedModelInfo(
            model_id="model2",
            model_type="lora",
            size_bytes=512,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/m2",
            is_active=False,
        )
        
        all_models = manager.list_models(include_inactive=True)
        self.assertEqual(len(all_models), 2)
        
        active_models = manager.list_models(include_inactive=False)
        self.assertEqual(len(active_models), 1)
        self.assertEqual(active_models[0]["model_id"], "model1")

    @patch("src.core.model_cache.snapshot_download")
    def test_download_model_disabled(self, mock_snapshot):
        mock_snapshot.return_value = "/tmp/cached/model"
        
        with patch.dict(os.environ, {"CACHE_ENABLED": "false"}):
            manager = ModelCacheManager(
                cache_dir=str(self.cache_dir),
                manifest_file=str(self.manifest_file),
            )
            
            result = manager.download_model(
                model_id="test/model",
                model_type="base",
                revision="main",
                token="test_token",
            )
            
            self.assertEqual(result, "/tmp/cached/model")
            mock_snapshot.assert_called_once()
            self.assertEqual(len(manager.manifest), 0)

    @patch("src.core.model_cache.snapshot_download")
    def test_download_model_new(self, mock_snapshot):
        test_cache_path = self.cache_dir / "downloaded_model"
        test_cache_path.mkdir(parents=True)
        (test_cache_path / "test_file.bin").write_bytes(b"test" * 256)
        
        mock_snapshot.return_value = str(test_cache_path)
        
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=1,
            manifest_file=str(self.manifest_file),
        )
        
        result = manager.download_model(
            model_id="test/model",
            model_type="base",
            revision="main",
            token="test_token",
        )
        
        self.assertEqual(result, str(test_cache_path))
        self.assertTrue(manager.is_model_cached("test/model", "main"))
        
        info = manager.get_model_info("test/model", "main")
        self.assertEqual(info.model_id, "test/model")
        self.assertEqual(info.model_type, "base")
        self.assertGreater(info.size_bytes, 0)
        self.assertTrue(info.is_active)

    @patch("src.core.model_cache.snapshot_download")
    def test_download_model_already_cached(self, mock_snapshot):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/existing",
            is_active=False,
        )
        
        result = manager.download_model(
            model_id="test/model",
            model_type="base",
            revision="main",
        )
        
        self.assertEqual(result, "/tmp/existing")
        mock_snapshot.assert_not_called()
        self.assertTrue(manager.manifest["test/model@main"].is_active)

    def test_evict_model(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        model_path = self.cache_dir / "test_model"
        model_path.mkdir(parents=True)
        (model_path / "file.bin").write_text("test data")
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path=str(model_path),
            is_active=False,
        )
        
        result = manager.evict_model("test/model", "main")
        
        self.assertTrue(result)
        self.assertFalse(manager.is_model_cached("test/model", "main"))
        self.assertFalse(model_path.exists())

    def test_evict_model_active_protection(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/test",
            is_active=True,
        )
        
        result = manager.evict_model("test/model", "main", force=False)
        
        self.assertFalse(result)
        self.assertTrue(manager.is_model_cached("test/model", "main"))

    def test_evict_model_force(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        model_path = self.cache_dir / "test_model"
        model_path.mkdir(parents=True)
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path=str(model_path),
            is_active=True,
        )
        
        result = manager.evict_model("test/model", "main", force=True)
        
        self.assertTrue(result)
        self.assertFalse(manager.is_model_cached("test/model", "main"))

    def test_auto_evict_lru_strategy(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=0.001,
            eviction_strategy=EvictionStrategy.LRU,
            manifest_file=str(self.manifest_file),
        )
        
        now = time.time()
        
        for i in range(3):
            model_path = self.cache_dir / f"model{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"model{i}@main"] = CachedModelInfo(
                model_id=f"model{i}",
                model_type="base",
                size_bytes=1024 * 1024,
                download_timestamp=now,
                last_accessed=now - (3 - i) * 100,
                revision="main",
                cache_path=str(model_path),
                is_active=False,
            )
        
        evicted = manager.auto_evict()
        
        self.assertGreater(evicted, 0)
        self.assertFalse(manager.is_model_cached("model0", "main"))

    def test_auto_evict_fifo_strategy(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=0.001,
            eviction_strategy=EvictionStrategy.FIFO,
            manifest_file=str(self.manifest_file),
        )
        
        now = time.time()
        
        for i in range(3):
            model_path = self.cache_dir / f"model{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"model{i}@main"] = CachedModelInfo(
                model_id=f"model{i}",
                model_type="base",
                size_bytes=1024 * 1024,
                download_timestamp=now + i * 10,
                last_accessed=now,
                revision="main",
                cache_path=str(model_path),
                is_active=False,
            )
        
        evicted = manager.auto_evict()
        
        self.assertGreater(evicted, 0)

    def test_auto_evict_size_strategy(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=0.001,
            eviction_strategy=EvictionStrategy.SIZE,
            manifest_file=str(self.manifest_file),
        )
        
        now = time.time()
        
        for i, size in enumerate([512, 2048, 1024]):
            model_path = self.cache_dir / f"model{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"model{i}@main"] = CachedModelInfo(
                model_id=f"model{i}",
                model_type="base",
                size_bytes=size * 1024,
                download_timestamp=now,
                last_accessed=now,
                revision="main",
                cache_path=str(model_path),
                is_active=False,
            )
        
        evicted = manager.auto_evict()
        
        self.assertGreater(evicted, 0)

    def test_auto_evict_respects_active_models(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=0.001,
            manifest_file=str(self.manifest_file),
        )
        
        now = time.time()
        
        for i in range(2):
            model_path = self.cache_dir / f"model{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"model{i}@main"] = CachedModelInfo(
                model_id=f"model{i}",
                model_type="base",
                size_bytes=1024 * 1024,
                download_timestamp=now,
                last_accessed=now,
                revision="main",
                cache_path=str(model_path),
                is_active=True,
            )
        
        evicted = manager.auto_evict()
        
        self.assertEqual(evicted, 0)
        self.assertEqual(len(manager.manifest), 2)

    def test_auto_evict_disabled_configuration(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            max_size_gb=0.001,
            auto_evict_enabled=False,
            manifest_file=str(self.manifest_file),
        )
        
        now = time.time()
        
        for i in range(2):
            model_path = self.cache_dir / f"model_disabled_{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"disabled{i}@main"] = CachedModelInfo(
                model_id=f"disabled{i}",
                model_type="base",
                size_bytes=1024 * 1024,
                download_timestamp=now,
                last_accessed=now,
                revision="main",
                cache_path=str(model_path),
                is_active=False,
            )
        
        evicted = manager.auto_evict()
        self.assertEqual(evicted, 0)
        self.assertEqual(len(manager.manifest), 2)
        
        evicted_forced = manager.auto_evict(force=True)
        self.assertGreater(evicted_forced, 0)

    def test_clear_cache(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        for i in range(3):
            model_path = self.cache_dir / f"model{i}"
            model_path.mkdir(parents=True)
            
            manager.manifest[f"model{i}@main"] = CachedModelInfo(
                model_id=f"model{i}",
                model_type="base",
                size_bytes=1024,
                download_timestamp=time.time(),
                last_accessed=time.time(),
                revision="main",
                cache_path=str(model_path),
                is_active=(i == 0),
            )
        
        cleared = manager.clear_cache(force=False)
        
        self.assertEqual(cleared, 2)
        self.assertTrue(manager.is_model_cached("model0", "main"))

    @patch("src.core.model_cache.snapshot_download")
    def test_request_load_model_cached(self, mock_snapshot):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/cached",
            is_active=False,
        )
        
        success, path = manager.request_load_model("test/model", "base", "main")
        
        self.assertTrue(success)
        self.assertEqual(path, "/tmp/cached")
        mock_snapshot.assert_not_called()
        self.assertTrue(manager.manifest["test/model@main"].is_active)

    def test_request_unload_model(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["test/model@main"] = CachedModelInfo(
            model_id="test/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/test",
            is_active=True,
        )
        
        result = manager.request_unload_model("test/model", "main")
        
        self.assertTrue(result)
        self.assertFalse(manager.manifest["test/model@main"].is_active)

    @patch("src.core.model_cache.snapshot_download")
    def test_prepare_lora_merge(self, mock_snapshot):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        manager.manifest["base/model@main"] = CachedModelInfo(
            model_id="base/model",
            model_type="base",
            size_bytes=1024,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="main",
            cache_path="/tmp/base",
            is_active=False,
        )
        
        manager.manifest["adapter/model@v1"] = CachedModelInfo(
            model_id="adapter/model",
            model_type="lora",
            size_bytes=512,
            download_timestamp=1.0,
            last_accessed=time.time(),
            revision="v1",
            cache_path="/tmp/adapter",
            is_active=False,
        )
        
        success, adapter_path, base_path = manager.prepare_lora_merge(
            "adapter/model", "base/model", "v1"
        )
        
        self.assertTrue(success)
        self.assertEqual(adapter_path, "/tmp/adapter")
        self.assertEqual(base_path, "/tmp/base")
        mock_snapshot.assert_not_called()

    def test_calculate_dir_size(self):
        manager = ModelCacheManager(
            cache_dir=str(self.cache_dir),
            manifest_file=str(self.manifest_file),
        )
        
        test_dir = self.cache_dir / "test_dir"
        test_dir.mkdir(parents=True)
        
        (test_dir / "file1.txt").write_bytes(b"x" * 100)
        (test_dir / "file2.txt").write_bytes(b"y" * 200)
        
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_bytes(b"z" * 50)
        
        size = manager._calculate_dir_size(str(test_dir))
        
        self.assertEqual(size, 350)


if __name__ == "__main__":
    unittest.main()
