import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # 3x96x96 --> 8x48x48
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.MaxPool2d(2))
        
        # 8x48x48 --> 16x24x24
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.MaxPool2d(2))
        
        # 16x24x24 --> 32x12x12
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.MaxPool2d(2))
        
        self.lin1 = nn.Sequential(nn.Linear(32*12*12 + 7, 1024),
                                  nn.LeakyReLU(negative_slope=0.2))
        
        self.lin2 = nn.Sequential(nn.Linear(1024, action_size))
        

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        observation = torch.Tensor(observation).to(self.device)
        batch_size = observation.shape[0]

        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)
        sensor_values = torch.cat((speed, abs_sensors, steering, gyroscope), dim=1) # Resulting shape: (batchsize, 7)

        x = torch.permute(observation, (0, 3, 1, 2))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat((torch.flatten(x, start_dim=1), sensor_values), dim=1)

        x = self.lin1(x)
        q_values = self.lin2(x)

        return q_values


    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels. The values are
        only approx. normalized, however, that suffices.
        Parameters
        ----------
        observation: list
            torch.Tensors of size (batch_size, 96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
