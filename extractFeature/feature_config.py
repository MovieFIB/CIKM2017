GPU_ID = 0

# True: resizes images to 448x448. False: resizes images to 224x224
USE_LARGE_INPUT_IMAGES = False

# True: flips images horizontally
FLIP_IMAGE = False

# These are in the repo
RESNET_LARGE_PROTOTXT_PATH = "./models/ResNet-152-448-deploy.prototxt"
RESNET_PROTOTXT_PATH = "./models/ResNet-152-deploy.prototxt"
RESNET_MEAN_PATH = "./models/ResNet_mean.binaryproto"

# Download caffemodel from https://github.com/KaimingHe/deep-residual-networks
RESNET_CAFFEMODEL_PATH = "./ResNet-152-model.caffemodel"

# COCO_IMAGE_PATH = "/data2/daylen/coco/images/"
# OUTPUT_PATH = "/data2/seth/vqa_test_res5c/"
# OUTPUT_PREFIX = "resnet_res5c_bgrms_large/"

# Which layer to extract and the size of the layer
EXTRACT_LAYER_POOL5 = "pool5"
EXTRACT_LAYER_CONV5C = "res5c"
EXTRACT_LAYER_SIZE = (2048,)
