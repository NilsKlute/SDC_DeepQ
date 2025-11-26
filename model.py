import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, device):
        """Create Noisy Linear Layer
        Parameters
        ---------
        dim_in: int
            the input dimension
        dim_out: int
            the ouptu dimension
        device: string
            device on which to the model will be allocated
        """
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device

        self.linear_layer_mean = nn.LinearLayer(dim_in, dim_out)
        self.linear_layer_sigma_w = torch.Parameter(torch.randn(1,1))
        self.linear_layer_sigma_b = torch.Parameter(torch.randn(1,1))

    def forward(self, x):
        mean = self.linear_layer_mean(x)

        noise_factors_in_dim = torch.randn(1, self.dim_in)
        noise_factors_out_dim = torch.randn(self.dim_out, 1)
        noise_weights = torch.matmul(noise_factors_out_dim, noise_factors_in_dim)
        noise_bias = noise_factors_out_dim

        out = mean + torch.matmul(self.linear_layer_sigma_w * noise_weights, x) + self.linear_layer_sigma_b * noise_bias

        return out



class DQN(nn.Module):
    def __init__(self, action_size, device, noisy=False):
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
        self.noisy = noisy

        
        # 3x96x96 --> 32x24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU()
        ) 

        # 32x24x24 --> 64×12x12
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )  

        # 64×12x12 --> 64×6x6
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        if noisy:
            #  noisy net linear layers 
            self.features =  nn.Sequential(NoisyLinearLayer(64*6*6 + 7, 512), nn.ReLU())

            self.value =  nn.Sequential(NoisyLinearLayer(512, 256), nn.ReLU(), NoisyLinearLayer(256, 1))

            self.advantage =  nn.Sequential(NoisyLinearLayer(512, 256), nn.ReLU(), NoisyLinearLayer(256, action_size))
        
        else:
            self.features = nn.Sequential(nn.Linear(64*6*6 + 7, 512), nn.ReLU())

            self.value = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
            
            self.advantage = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, action_size))

        

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

        features = self.features(x)

        V = self.value(features)
        A = self.advantage(features)

        q_values = V + (A - A.mean(dim=1, keepdim=True))

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
