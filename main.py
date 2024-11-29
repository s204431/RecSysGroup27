import torch
import torch.nn as nn
from UserEncoder import UserEncoder
import os

user_encoder = UserEncoder(4, 0.2)
test_name = 'blabla'

def main():

    for epoch in range(1, 3):  # Eksempel med epoch fra 1 til 2
        """ base_filename = f"user_encoder_{test_name}_{epoch}.pth"
        filename = base_filename
        counter = 1

        # Tjek, om filen eksisterer, og find et ledigt navn
        while os.path.exists(filename):
            filename = f"user_encoder_{test_name}_{epoch}_v{counter}.pth"
            counter += 1

        torch.save(user_encoder.state_dict(), filename)
        print(f"Model gemt som: {filename}") """

        filename = f"Models/user_encoder_{test_name}_{epoch}.pth"
        torch.save(user_encoder.state_dict(), filename)
        print(f"Model gemt som: {filename}")



























if __name__ == '__main__':
    main()