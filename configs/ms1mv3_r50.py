from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512
config.lr = 0.01  # batch size is 512

config.rec = "/home/dmitriy/develop/oz_forensics/datasets"
config.num_classes = 10572
config.num_image = 490623
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["lfw"]
