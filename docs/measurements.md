# Measurement notes

The checked-in example is a real CPU LoRA run against the pinned Google BERT Tiny checkpoint. The base encoder is pretrained; the classification head is initialized for the six support-intent labels.

The dataset is synthetic, with eight template families per label. Five variants of each template produce 240 examples. Template groups are separated into 180 training, 30 validation, and 30 test examples. The reference run selected epoch 4 using validation macro F1, then reloaded that checkpoint for test evaluation.

The test macro F1 was **0.519943**, with accuracy **0.566667**. The before-adaptation classifier had macro F1 **0.047619**. This baseline has an untrained head; a head-only recipe is the appropriate separate experiment for measuring the benefit of encoder adapters over classifier training alone.

The run updated **8,966 of 4,395,660 wrapped-model parameters**, or **0.2040%**. It used rank-8 query/value adapters, learning rate 0.001, batch size 8, sequence length 96, four epochs, and seed 42. The worker recorded 92 optimizer steps and 164.736 seconds elapsed, including model preparation and evaluation, on an x86-64 CPU with two PyTorch threads. This is a workspace observation, not a controlled hardware benchmark.

The 30-example test split has only five examples per class. Template wording and generated identifiers make it unsuitable for claims about real support traffic. No production accuracy, comparative state-of-the-art result, or calibrated confidence estimate is claimed.

The files in `example-run/` preserve the actual run metadata, test measurements, dataset manifest, and loss history. Model weights remain in the local artifact store and are not checked into Git. Export/reload tests verify that trained adapters restore the classifier as well as adapter tensors.
