# Local-SGD Implementation in PolarParallel

## 📖 Overview

Local-SGD (Local Stochastic Gradient Descent) is an optimization algorithm for distributed training that **reduces communication frequency** by synchronizing model **parameters** instead of **gradients** at every step.

### Key Differences from Standard Data Parallel Training

| Aspect | Standard DP | Local-SGD |
|--------|-------------|-----------|
| **Sync target** | Gradients | Parameters |
| **Sync frequency** | Every step | Every N steps |
| **Communication volume** | High | Lower (1/N) |
| **Convergence** | Faster (more sync) | Slightly slower but acceptable |
| **Scalability** | Limited by comm | Better for slow networks |

---

## 🎯 When to Use Local-SGD

✅ **Good for:**
- Slow inter-node networks (high latency, low bandwidth)
- Large models with many parameters
- Scenarios where communication is the bottleneck
- Multi-node training with limited interconnect

❌ **Not ideal for:**
- Single-node multi-GPU training (fast NVLink/PCIe)
- Small models where computation is the bottleneck
- Tasks requiring very frequent synchronization

---

## 🚀 Usage

### Command Line Arguments

```bash
python tests/train_llama7b_polar_dp_pp.py \
  --use_local_sgd \              # Enable Local-SGD mode
  --local_sgd_steps 10 \          # Sync parameters every 10 steps
  # ...other args...
```

### Quick Start Scripts

#### Basic Local-SGD Training (sync every 10 steps)
```bash
# Node 0
bash scripts/2dp_8pp/0_train_llama7b_local_sgd.sh

# Node 1
bash scripts/2dp_8pp/1_train_llama7b_local_sgd.sh
```

#### Custom Sync Interval
```bash
# Sync every 20 steps for even lower communication
LOCAL_SGD_STEPS=20 bash scripts/2dp_8pp/0_train_llama7b_local_sgd.sh
```

#### Comparison Experiment
```bash
# Compare different sync intervals (5, 10, 20 steps vs baseline)
bash scripts/experiments/compare_local_sgd_intervals.sh
```

---

## 🔧 Technical Details

### Implementation

The Local-SGD implementation in `PolarParallel` includes:

1. **Parameter Synchronization Method** (`_sync_parameters_local_sgd`):
   ```python
   def _sync_parameters_local_sgd(self):
       """Average parameters across all DP replicas."""
       for param in self.stage.submod.parameters():
           dist.all_reduce(param.data, op=SUM, group=dp_group)
           param.data /= dp_size
   ```

2. **Training Loop Integration**:
   - Each worker updates parameters locally using its own data
   - Every `N` steps, all workers synchronize by averaging their parameters
   - Between syncs, workers maintain independent parameter values

3. **Compatibility**:
   - Works with both DP and PP (Pipeline Parallel)
   - Compatible with different optimizers (AdamW, SGD)
   - Can be combined with gradient accumulation

### Algorithm

```
for step in training_steps:
    # 1. Forward pass with local batch
    loss = model(batch)
    
    # 2. Backward pass (compute gradients)
    loss.backward()
    
    # 3. Local optimizer step (NO gradient sync)
    optimizer.step()
    
    # 4. Parameter sync every N steps
    if step % local_sgd_steps == 0:
        sync_parameters()  # All-reduce parameters
```

---

## 📊 Expected Performance

### Communication Reduction

With `local_sgd_steps=10`:
- **Communication volume**: 10× reduction
- **Gradient sync**: 0 times per 10 steps
- **Parameter sync**: 1 time per 10 steps

### Convergence Trade-off

- **Sync interval = 1** (standard): Fastest convergence, highest comm cost
- **Sync interval = 5-10**: Good balance for most scenarios
- **Sync interval = 20+**: Lowest comm cost, may affect convergence

---

## 🔬 Experimental Setup

### Recommended Configurations

| Network Type | Recommended `local_sgd_steps` |
|--------------|------------------------------|
| InfiniBand (fast) | 1-5 |
| 10GbE (moderate) | 10-20 |
| 1GbE (slow) | 20-50 |
| Cross-datacenter | 50-100 |

### Hyperparameter Tuning

When using Local-SGD, consider:
- **Increase learning rate** slightly (1.2-1.5×) to compensate for delayed sync
- **Use warmup** to stabilize training in early stages
- **Monitor gradient staleness**: if loss diverges, reduce `local_sgd_steps`

---

## 📈 Monitoring

Check TensorBoard logs to compare:
- **Training loss convergence**
- **Throughput (samples/sec)**
- **Communication time (profiler)**

```bash
tensorboard --logdir ./log/
```

---

## 🛠️ Implementation Notes

### Differences from Polar Gradient Prediction

| Feature | Polar (Gradient Prediction) | Local-SGD |
|---------|----------------------------|-----------|
| **Sync type** | Predicted gradients | Parameters |
| **Hook usage** | Yes (GpipeHook) | No (direct sync) |
| **Communication pattern** | Async overlap | Periodic sync |
| **Memory overhead** | Higher (pred buffers) | Lower |

### Disabling Hooks in Local-SGD Mode

When `use_local_sgd=True`:
- Polar gradient prediction hooks are **disabled**
- No gradient communication occurs
- Only parameter averaging happens at sync points

---

## 📚 References

1. Lin, T., et al. "Don't Use Large Mini-Batches, Use Local SGD." ICLR 2020.
2. Stich, S. U. "Local SGD Converges Fast and Communicates Little." ICLR 2019.
3. Yu, H., et al. "Parallel Restarted SGD with Faster Convergence and Less Communication." NeurIPS 2019.

---

## 🐛 Troubleshooting

### Loss Divergence
- **Reduce** `local_sgd_steps` (try 5 instead of 20)
- **Lower** learning rate
- **Increase** batch size per worker

### Low Speedup
- **Increase** `local_sgd_steps` if network is slow
- **Check** if computation is bottleneck (profile with TensorBoard)

### Parameter Mismatch Errors
- Ensure all workers use **same** `local_sgd_steps` value
- Verify DP group is correctly initialized

---

For questions or issues, please check the main README or open an issue.
