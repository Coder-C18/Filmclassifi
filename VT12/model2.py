import torch
import torch.nn as nn
import torchvision


class ClassifyFilm(nn.Module):
    def __init__(self):
        super(ClassifyFilm, self).__init__()

        self.descriptions = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2))
        self.actor = nn.Sequential(nn.Linear(713, 512),
                                   nn.ReLU())

        self.director = nn.Sequential(nn.Linear(786, 512),
                                      nn.ReLU())
        self.flatten = nn.Flatten()
        self.classify = nn.Sequential(nn.Linear(2048, 1024),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(1024, 512),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(256, 128),
                                      nn.LeakyReLU(0.2),
                                      )

        self.fc1 = nn.Sequential(nn.Linear(128, 23),
                                 nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(128,64),
                                 nn.ReLU(),
                                 nn.Linear(64,32),
                                 nn.ReLU(),
                                 nn.Linear(32, 7),
                                 nn.Softmax(dim=1))

    def forward(self, images, actor, director, description):
            actor_emb = self.actor(actor)
            director_emb = self.director(director)
            feature1 = torch.cat([actor_emb, director_emb], dim=2)

            feature2 = torch.unsqueeze(description, dim=1)
            feature2 = self.descriptions(feature2)

            feature12 = torch.concat([feature1, feature2], dim=1)

            feature_text = self.flatten(feature12)

            last = self.classify(feature_text)

            out1 = self.fc1(last)
            out2 = self.fc2(last)

            return out1, out2
