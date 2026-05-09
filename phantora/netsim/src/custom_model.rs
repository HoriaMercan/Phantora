use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModelFeature {
    pub feature: String,
    pub coefficient: f64,
    #[serde(default)]
    pub importance: Option<f64>,
    #[serde(default)]
    pub std: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModelOperation {
    pub intercept: f64,
    #[serde(default)]
    pub selected_features: Vec<CustomModelFeature>,
}

#[derive(Debug, Clone)]
enum OperationModel {
    Single(CustomModelOperation),
    Piecewise {
        threshold: usize,
        low_model: CustomModelOperation,
        high_model: CustomModelOperation,
    },
}

impl OperationModel {
    fn operation_for_size(&self, size: usize) -> &CustomModelOperation {
        match self {
            OperationModel::Single(model) => {
                log::trace!("Using single model for size {}", size);
                model
            }
            OperationModel::Piecewise {
                threshold,
                low_model,
                high_model,
            } => {
                if size < *threshold {
                    log::debug!("Size {} < threshold {}, using low_model", size, threshold);
                    low_model
                } else {
                    log::debug!("Size {} >= threshold {}, using high_model", size, threshold);
                    high_model
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomModelListEntry {
    #[serde(default)]
    callpath: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    piecewise: bool,
    #[serde(default)]
    memory_threshold: Option<f64>,
    #[serde(default)]
    low_model: Option<CustomModelOperation>,
    #[serde(default)]
    high_model: Option<CustomModelOperation>,
    #[serde(default)]
    intercept: Option<f64>,
    #[serde(default)]
    selected_features: Option<Vec<CustomModelFeature>>,
}

impl CustomModelListEntry {
    fn to_operation_model(&self) -> Option<OperationModel> {
        if self.piecewise {
            let threshold = self.memory_threshold?;
            let low_model = self.low_model.clone()?;
            let high_model = self.high_model.clone()?;
            return Some(OperationModel::Piecewise {
                threshold: threshold.max(0.0) as usize,
                low_model,
                high_model,
            });
        }

        let intercept = self.intercept?;
        let selected_features = self.selected_features.clone().unwrap_or_default();
        Some(OperationModel::Single(CustomModelOperation {
            intercept,
            selected_features,
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModelConfig {
    #[serde(skip)]
    models: HashMap<String, OperationModel>,
    
    #[serde(skip)]
    pub bw_mbps: f64,
    #[serde(skip)]
    pub lacking_nodes: f64,
    #[serde(skip)]
    pub default_latency_us: f64,
}

impl CustomModelConfig {
    pub fn load_from_path(
        base_path: &Path,
        bw_mbps: f64,
        lacking_nodes: f64,
        default_latency_us: f64,
        topology: Option<&str>,
    ) -> Option<Self> {
        log::info!("Loading custom model from path: {:?}, topology: {:?}", base_path, topology);
        let mut models = HashMap::new();
        let normalized_topology = topology.map(Self::normalize_topology);
        
        let dir = if base_path.is_dir() {
            base_path.to_path_buf()
        } else {
            // Also try to load the base_path directly if it's a file
            if let Ok(content) = fs::read_to_string(base_path) {
                log::debug!("Loading custom model from file: {:?}", base_path);
                let file_stem = base_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                Self::load_models_from_json_content(
                    &mut models,
                    &content,
                    file_stem,
                    normalized_topology.as_deref(),
                );
            }
            base_path.parent()?.to_path_buf()
        };
        
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        log::debug!("Loading custom model from JSON: {:?}", path);
                        let file_stem = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or_default();
                        Self::load_models_from_json_content(
                            &mut models,
                            &content,
                            file_stem,
                            normalized_topology.as_deref(),
                        );
                    }
                }
            }
        }
        
        log::info!("Custom model loaded with {} model entries", models.len());
        Some(CustomModelConfig {
            models,
            bw_mbps,
            lacking_nodes,
            default_latency_us,
        })
    }

    fn normalize_key(key: &str) -> String {
        key.chars()
            .flat_map(|c| c.to_lowercase())
            .collect()
    }

    fn normalize_topology(topology: &str) -> String {
        let normalized: String = topology
            .chars()
            .flat_map(|c| c.to_lowercase())
            .collect();

        if normalized.contains("dragonfly") {
            return "dragonfly".to_string();
        }
        if normalized.contains("fattree") || normalized.contains("fatree") {
            return "fattree".to_string();
        }
        if normalized.contains("torus") {
            return "torus".to_string();
        }

        normalized
    }

    fn topology_from_file_path(path: &str) -> Option<String> {
        let mut parts = path.split('/');
        let _root = parts.next();
        let topology = parts.next()?;
        Some(Self::normalize_topology(topology))
    }

    fn insert_model(models: &mut HashMap<String, OperationModel>, key: &str, model: OperationModel) {
        models.insert(Self::normalize_key(key), model);
    }

    fn operation_aliases(operation: &str) -> Vec<String> {
        let mut aliases = vec![operation.to_string()];
        let normalized = operation.replace('_', "");
        if normalized != operation {
            aliases.push(normalized.clone());
        }

        match normalized.as_str() {
            "allreduce" => aliases.push("all_reduce".to_string()),
            "alltoall" => aliases.push("all_to_all".to_string()),
            "allgather" => aliases.push("all_gather".to_string()),
            "broadcast" | "bcast" => {
                aliases.push("broadcast".to_string());
                aliases.push("bcast".to_string());
            }
            "reducescatter" => aliases.push("reduce_scatter".to_string()),
            _ => {}
        }

        aliases.sort();
        aliases.dedup();
        aliases
    }

    fn load_models_from_json_content(
        models: &mut HashMap<String, OperationModel>,
        content: &str,
        file_stem: &str,
        topology_filter: Option<&str>,
    ) {
        if let Ok(entries) = serde_json::from_str::<Vec<CustomModelListEntry>>(content) {
            log::debug!("Loaded {} entries from JSON array", entries.len());
            for (idx, entry) in entries.iter().enumerate() {
                log::debug!("Entry {}: callpath={:?}, file={:?}, piecewise={}", idx, entry.callpath, entry.file, entry.piecewise);
                
                // Only filter by topology if both filter and file path are present
                if let (Some(filter), Some(file_path)) = (topology_filter, entry.file.as_deref()) {
                    if let Some(entry_topology) = Self::topology_from_file_path(file_path) {
                        if entry_topology != filter {
                            log::debug!("  Skipping Entry {}: topology {} != filter {}", idx, entry_topology, filter);
                            continue;
                        } else {
                            log::debug!("  Entry {} matches topology filter: {}", idx, filter);
                        }
                    }
                } else if topology_filter.is_some() {
                    log::debug!("  Entry {} has no file path, loading anyway (no topology check)", idx);
                }

                if let Some(op_model) = entry.to_operation_model() {
                    if let Some(callpath) = entry.callpath.as_deref() {
                        log::debug!("  Inserting model for callpath: '{}', piecewise={}", callpath, entry.piecewise);
                        Self::insert_model(models, callpath, op_model);
                    } else {
                        // Entry has no callpath - use as default model for all standard NCCL operations
                        log::debug!("  Entry {} has no callpath, registering as default model", idx);
                        Self::insert_model(models, "nccl_all_reduce", op_model.clone());
                        Self::insert_model(models, "nccl_allreduce", op_model.clone());
                        Self::insert_model(models, "nccl_broadcast", op_model.clone());
                        Self::insert_model(models, "nccl_bcast", op_model.clone());
                        Self::insert_model(models, "nccl_allgather", op_model.clone());
                        Self::insert_model(models, "nccl_alltoall", op_model.clone());
                        Self::insert_model(models, "nccl_reduce", op_model.clone());
                        Self::insert_model(models, "nccl_reduce_scatter", op_model);
                    }
                } else {
                    log::debug!("  Failed to create operation model from entry {}", idx);
                }
            }
            return;
        }

        if let Ok(entry) = serde_json::from_str::<CustomModelListEntry>(content) {
            log::debug!("Loaded single entry from JSON: callpath={:?}", entry.callpath);
            if let Some(op_model) = entry.to_operation_model() {
                if let Some(callpath) = entry.callpath.as_deref() {
                    Self::insert_model(models, callpath, op_model);
                } else if !file_stem.is_empty() {
                    Self::insert_model(models, file_stem, op_model);
                }
                return;
            }
        }

        // Try parsing as array of raw CustomModelOperation objects
        if let Ok(ops) = serde_json::from_str::<Vec<CustomModelOperation>>(content) {
            log::debug!("Loaded {} raw CustomModelOperation entries from JSON array", ops.len());
            if !ops.is_empty() {
                let op_model = OperationModel::Single(ops[0].clone());
                if !file_stem.is_empty() {
                    Self::insert_model(models, file_stem, op_model.clone());
                }
                // fallback if there's only one operation file provided
                Self::insert_model(models, "nccl_all_reduce", op_model.clone());
                Self::insert_model(models, "nccl_allreduce", op_model.clone());
                Self::insert_model(models, "nccl_broadcast", op_model.clone());
                Self::insert_model(models, "nccl_bcast", op_model);
            }
            return;
        }

        if let Ok(op) = serde_json::from_str::<CustomModelOperation>(content) {
            log::debug!("Loaded raw CustomModelOperation from file_stem: '{}'", file_stem);
            let op_model = OperationModel::Single(op.clone());
            if !file_stem.is_empty() {
                Self::insert_model(models, file_stem, op_model.clone());
            }

            // fallback if there's only one operation file provided
            Self::insert_model(models, "nccl_all_reduce", op_model.clone());
            Self::insert_model(models, "nccl_allreduce", op_model.clone());
            Self::insert_model(models, "nccl_broadcast", op_model.clone());
            Self::insert_model(models, "nccl_bcast", op_model);
        }
    }

    fn evaluate_term(term: &str, size: f64, nranks: f64, bw_mbps: f64, lacking_nodes: f64, latency_us: f64) -> f64 {
        let term = term.trim();
        if term == "message_size" || term == "size" {
            return size;
        }
        if term == "nodes" {
            return nranks;
        }
        if term == "log2(nodes)" {
            return nranks.log2();
        }
        if term == "1/bw_mbps" {
            return 1.0 / bw_mbps;
        }
        if term == "bw_mbps" {
            return bw_mbps;
        }
        if term == "lacking_nodes" {
            return lacking_nodes;
        }
        if term == "latency_us" {
            return latency_us;
        }
        
        if term.starts_with("nodes^") {
            let (_, pow_str) = term.split_at(6);
            let mut pow_val = 1.0;
            if pow_str.starts_with("(") && pow_str.ends_with(")") {
                let inner = &pow_str[1..pow_str.len()-1];
                let parts: Vec<&str> = inner.split('/').collect();
                if parts.len() == 2 {
                    if let (Ok(n), Ok(d)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                        pow_val = n / d;
                    }
                } else if let Ok(v) = inner.parse::<f64>() {
                    pow_val = v;
                }
            } else if let Ok(v) = pow_str.parse::<f64>() {
                pow_val = v;
            }
            return nranks.powf(pow_val);
        }
        
        1.0
    }

    fn evaluate_feature(feature_str: &str, size: usize, nranks: usize, bw_mbps: f64, lacking_nodes: f64, latency_us: f64) -> f64 {
        let size_f = size as f64;
        let nranks_f = nranks as f64;
        let mut val = 1.0;
        for part in feature_str.split('*') {
            val *= Self::evaluate_term(part, size_f, nranks_f, bw_mbps, lacking_nodes, latency_us);
        }
        val
    }

    pub fn estimate_time(&self, framework: &str, operation: &str, size: usize, nranks: usize) -> Option<f64> {
        // If size > 0.5GB, call the functino for size 500MB and scale the result linearly with size. This is to avoid extrapolating too much beyond the training data.
        if size > 500_000 { // 0.5GB in bytes
            // let base_time = self.estimate_time(framework, operation, 500_000_000, nranks)?;
            // let scaled_time = base_time * (size as f64 / 500_000.0);
            // log::debug!("Size {} exceeds 0.5GB, scaling estimated time {} by factor {}", size, base_time, size as f64 / 500_000_000.0);
            // return Some(scaled_time);
        }
        
        let mut keys_to_try = Vec::new();
        for op_alias in Self::operation_aliases(operation) {
            keys_to_try.push(format!("{}_{}", framework, op_alias));
            keys_to_try.push(format!("custom_model_results_{}_{}", framework, op_alias));
            keys_to_try.push(op_alias.clone());
            keys_to_try.push(format!("nccl_{}", op_alias));
        }

        log::debug!("Keys to try (before normalization): {:?}", keys_to_try);

        let mut op_model = None;
        for k in &keys_to_try {
            let normalized = Self::normalize_key(&k);
            log::debug!("Trying key '{}' normalized to '{}'", k, normalized);
            if let Some(m) = self.models.get(&normalized) {
                log::debug!("Found model for key '{}'", normalized);
                op_model = Some(m.operation_for_size(size));
                break;
            }
        }

        let op_model = op_model?;

        let mut latency = op_model.intercept;
        for feat in &op_model.selected_features {
            let feat_val = Self::evaluate_feature(
                &feat.feature,
                if size > 500_000 { 500_000 } else { size }, // cap size for feature evaluation to avoid extreme extrapolation
                nranks,
                self.bw_mbps,
                self.lacking_nodes,
                self.default_latency_us
            );
            latency += feat.coefficient * feat_val;
        }

        if latency < 0.0 {
            latency = 0.0;
        }

        Some(latency)
    }
}

#[cfg(test)]
mod tests {
    use super::CustomModelConfig;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn write_temp_json(name: &str, content: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("{}_{}_{}.json", name, std::process::id(), nanos));
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn loads_piecewise_entry_and_selects_by_message_size() {
        let content = r#"[
            {
                "callpath": "nccl_allreduce",
                "file": "extrap-gpu/torus/allreduce.json",
                "piecewise": true,
                "memory_threshold": 1024,
                "low_model": {
                    "intercept": 10.0,
                    "selected_features": []
                },
                "high_model": {
                    "intercept": 20.0,
                    "selected_features": []
                }
            }
        ]"#;
        let path = write_temp_json("custom_model_piecewise", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, Some("torus"))
            .expect("failed to load piecewise model");

        assert_eq!(cfg.estimate_time("nccl", "all_reduce", 512, 8), Some(10.0));
        assert_eq!(cfg.estimate_time("nccl", "all_reduce", 2048, 8), Some(20.0));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn filters_entries_by_topology() {
        let content = r#"[
            {
                "callpath": "nccl_bcast",
                "file": "extrap-gpu/fattree/bcast.json",
                "intercept": 7.0,
                "selected_features": []
            }
        ]"#;
        let path = write_temp_json("custom_model_topology", content);

        let cfg = CustomModelConfig::load_from_path(
            Path::new("/home/horia/Desktop/Facultate/Anul4/Phantora/custom_model_results.json"),
             100_000.0, 0.0, 1.0, Some("torus"))
            .expect("failed to load model");

        println!("Loaded models: {:?}", cfg.models.keys().collect::<Vec<_>>());

        assert_ne!(cfg.models.len(), 0, "models should be loaded");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_entry_without_callpath_as_default_model() {
        let content = r#"[
            {
                "intercept": 100.0,
                "selected_features": [
                    {
                        "feature": "message_size",
                        "coefficient": 0.5,
                        "importance": 80.0,
                        "std": 10.0
                    },
                    {
                        "feature": "log2(nodes)",
                        "coefficient": 50.0,
                        "importance": 20.0,
                        "std": 5.0
                    }
                ]
            }
        ]"#;
        let path = write_temp_json("custom_model_default", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, None)
            .expect("failed to load default model");

        // Default model should be registered for standard NCCL operations
        assert_eq!(cfg.estimate_time("nccl", "allreduce", 1024, 8), Some(100.0 + 0.5*1024.0 + 50.0*3.0)); // log2(8) = 3
        assert_eq!(cfg.estimate_time("nccl", "broadcast", 512, 4), Some(100.0 + 0.5*512.0 + 50.0*2.0)); // log2(4) = 2
        assert_eq!(cfg.estimate_time("nccl", "allgather", 256, 16), Some(100.0 + 0.5*256.0 + 50.0*4.0)); // log2(16) = 4

        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_entry_with_callpath_overrides_default() {
        let content = r#"[
            {
                "intercept": 10.0,
                "selected_features": []
            },
            {
                "callpath": "nccl_allreduce",
                "intercept": 20.0,
                "selected_features": [
                    {
                        "feature": "message_size",
                        "coefficient": 1.0,
                        "importance": 100.0,
                        "std": 1.0
                    }
                ]
            }
        ]"#;
        let path = write_temp_json("custom_model_override", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, None)
            .expect("failed to load model");

        // Default model for broadcast
        assert_eq!(cfg.estimate_time("nccl", "broadcast", 512, 4), Some(10.0));
        // Specific model for allreduce should use the override
        assert_eq!(cfg.estimate_time("nccl", "allreduce", 512, 4), Some(20.0 + 1.0*512.0));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn estimate_time_with_complex_features() {
        let content = r#"[
            {
                "intercept": 50.0,
                "selected_features": [
                    {
                        "feature": "message_size * log2(nodes)",
                        "coefficient": 0.1,
                        "importance": 50.0,
                        "std": 5.0
                    },
                    {
                        "feature": "nodes^(1/2)",
                        "coefficient": 10.0,
                        "importance": 30.0,
                        "std": 3.0
                    },
                    {
                        "feature": "1/bw_mbps * message_size",
                        "coefficient": 0.001,
                        "importance": 20.0,
                        "std": 2.0
                    }
                ]
            }
        ]"#;
        let path = write_temp_json("custom_model_complex", content);

        let cfg = CustomModelConfig::load_from_path(&path, 50_000.0, 0.0, 1.0, None)
            .expect("failed to load model");

        // Test with size=256, nranks=16
        // log2(16) = 4
        // 256 * 4 = 1024
        // sqrt(16) = 4
        // 1/50000 = 0.00002
        // Expected: 50 + 0.1*1024 + 10*4 + 0.001*0.00002*256
        let result = cfg.estimate_time("nccl", "allreduce", 256, 16).expect("estimate failed");
        let expected = 50.0 + 0.1*1024.0 + 10.0*4.0 + 0.001*0.00002*256.0;
        assert!((result - expected).abs() < 1e-6, "got {}, expected {}", result, expected);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn estimate_time_clamps_negative_values_to_zero() {
        let content = r#"[
            {
                "intercept": -1000.0,
                "selected_features": []
            }
        ]"#;
        let path = write_temp_json("custom_model_negative", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, None)
            .expect("failed to load model");

        let result = cfg.estimate_time("nccl", "allreduce", 512, 4).expect("estimate failed");
        assert_eq!(result, 0.0, "negative estimates should be clamped to 0");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn piecewise_model_selects_by_message_size() {
        let content = r#"[
            {
                "callpath": "nccl_reduce",
                "piecewise": true,
                "memory_threshold": 4096.0,
                "low_model": {
                    "intercept": 5.0,
                    "selected_features": [
                        {
                            "feature": "message_size",
                            "coefficient": 0.01,
                            "importance": 100.0,
                            "std": 1.0
                        }
                    ]
                },
                "high_model": {
                    "intercept": 50.0,
                    "selected_features": [
                        {
                            "feature": "log2(nodes)",
                            "coefficient": 20.0,
                            "importance": 100.0,
                            "std": 1.0
                        }
                    ]
                }
            }
        ]"#;
        let path = write_temp_json("custom_model_piecewise_advanced", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, None)
            .expect("failed to load model");

        // Below threshold: use low_model
        let small = cfg.estimate_time("nccl", "reduce", 1024, 8).expect("estimate failed");
        assert_eq!(small, 5.0 + 0.01*1024.0);

        // Above threshold: use high_model
        let large = cfg.estimate_time("nccl", "reduce", 8192, 8).expect("estimate failed");
        assert_eq!(large, 50.0 + 20.0*3.0); // log2(8) = 3

        let _ = fs::remove_file(path);
    }

    #[test]
    fn operation_aliases_work_correctly() {
        let content = r#"[
            {
                "callpath": "all_reduce",
                "intercept": 10.0,
                "selected_features": []
            }
        ]"#;
        let path = write_temp_json("custom_model_aliases", content);

        let cfg = CustomModelConfig::load_from_path(&path, 100_000.0, 0.0, 1.0, None)
            .expect("failed to load model");

        // Should be findable under all aliases
        assert_eq!(cfg.estimate_time("nccl", "all_reduce", 512, 4), Some(10.0));
        assert_eq!(cfg.estimate_time("nccl", "allreduce", 512, 4), Some(10.0));
        assert_eq!(cfg.estimate_time("nccl", "AllReduce", 512, 4), Some(10.0));

        let _ = fs::remove_file(path);
    }
}