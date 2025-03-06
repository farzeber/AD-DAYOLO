import os
import time

import torch
import torchvision

bs = 36
print(torch.cuda.device_count())
test_img = torch.Tensor(bs, 3, 224, 224).cuda()
model = torchvision.models.vgg16(pretrained=False,
                                     init_weights=True).cuda()
model.eval()

while True:
    time.sleep(5)
    torch.cuda.synchronize()  # 等待同步
    sta = time.time()
    result = model(test_img)
    torch.cuda.synchronize()  # 等待同步
    end = time.time()
    print(f"- Pytorch forward time cost: {end - sta}")

    restart = './1.key'
    if os.path.exists(restart):
        time.sleep(2)
        if os.path.exists(restart):
            os.remove(restart)
            exit()
