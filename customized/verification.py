import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import pandas as pd
from datetime import datetime


class Verification:

    def __init__(self, data_dir, output_dir, run_name):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.run_name = run_name

    def verify(self, model, epoch):
        with open(os.path.join(self.data_dir, 'verification_pairs_test.txt'), 'r') as fd:
            content = fd.readlines()

        composition = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.229, 0.224, 0.225), std=(0.485, 0.456, 0.406))])
        compute_sim = nn.CosineSimilarity(dim=0)

        # for each pair of images in the test file
        results = []
        for line in content[:5]:
            file_a, file_b = line.strip().split()

            # read in images
            img_a = Image.open(os.path.join(self.data_dir, file_a))
            img_a = composition(img_a).unsqueeze(0)

            img_b = Image.open(os.path.join(self.data_dir, file_b))
            img_b = composition(img_b).unsqueeze(0)

            #     # move to device
            #     img_a.to(torch.device('cuda'))
            #     img_b.to(torch.device('cuda'))
            # move model to cpu
            model.eval()
            model = model.cpu()

            # send each image through model and get embedding
            embedding_a, out_a = model.forward(img_a, return_embedding=True)
            embedding_b, out_b = model.forward(img_b, return_embedding=True)

            # calculate similarity
            feats_a = embedding_a.squeeze(0)
            feats_b = embedding_b.squeeze(0)
            sim = compute_sim(feats_a, feats_b)
            print(sim.item())
            # store entry
            results.append([file_a + " " + file_b, sim.item()])

        # build dataframe of the results
        df = pd.DataFrame(results, columns=['Id', 'Category'])

        # save results to file
        filename = f'{self.run_name}.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.similarity.csv'
        df.to_csv(os.path.join(self.output_dir, filename), header=True, index=False)
