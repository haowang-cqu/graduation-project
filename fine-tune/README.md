# 微调下游模型评估攻击效果

### GLUE Benchmark
修改 `glue/glue.sh` 中的 `TASK_NAME` 和其他超参数
```bash
cd glue
sh glue.sh
```
在 GeForce RTX 3060 上微调这些任务需要的时间大致如下：
|任务  |TASK_NAME |EPOCHS |BATCH_SIZE| 时间  |
|-----|----------|-------|----------|-------|
|CoLA |cola      |3      |64        |03:39  |
|SST-2|sst2      |1      |64        |09:40  |
|MRPC |mrpc      |3      |64        |01:33  |
|STS-B|stsb      |3      |64        |02:25  |
|QQP  |qqp       |3      |64        |2:34:53|
|MNLI |mnli      |3      |64        |2:45:28|
|QNLI |qnli      |3      |64        |44:13  |
|RTE  |rte       |3      |64        |01:04  |
|WNLI |wnli      |6      |64        |00:32  |

注意：表中的时间是指微调一次所需要的时间，而运行一次脚本需要分别微调干净模型和后门模型，所以大概需要两倍于表中的时间。
