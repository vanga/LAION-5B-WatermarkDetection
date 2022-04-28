# LAION-5B-WatermarkDetection

Download the model from [releases](https://github.com/LAION-AI/LAION-5B-WatermarkDetection/releases) and place it at the root of this repository.

Sample usage
```
python main.py --urls="pipe:aws s3 cp --quiet s3://<s3-base-path>/humans-{000000..000001}.tar" --batch-size=512 --num-samples=1024
```