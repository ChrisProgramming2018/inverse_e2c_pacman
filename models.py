import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim):
        super(Transition, self).__init__()
        self.net = net  # network to output the last layer before predicting A_t, B_t and o_t
        self.net.apply(weights_init)
        self.h_dim = self.net[-3].out_features
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.fc_A = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim * 2), # v_t and r_t
            nn.Sigmoid()
        )
        self.fc_A.apply(weights_init)

        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(self.h_dim, self.z_dim)
        torch.nn.init.orthogonal_(self.fc_o.weight)

    def forward(self, z_bar_t, q_z_t, u_t):
        """
        :param z_bar_t: the reference point
        :param Q_z_t: the distribution q(z|x)
        :param u_t: the action taken
        :return: the predicted q(z^_t+1 | z_t, z_bar_t, u_t)
        """
        h_t = self.net(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)

        v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
        v_t = torch.unsqueeze(v_t, dim=-1)
        r_t = torch.unsqueeze(r_t, dim=-2)

        A_t = torch.eye(self.z_dim).repeat(z_bar_t.size(0), 1, 1).cuda() + torch.bmm(v_t, r_t)

        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        mu_t = q_z_t.mean

        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t

        return mean, NormalDistribution(mean, logvar=q_z_t.logvar, v=v_t.squeeze(), r=r_t.squeeze(), A=A_t)


class PacmanTransition(Transition):
    def __init__(self, z_dim = 3, u_dim = 1):
        net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        super(PacmanTransition, self).__init__(net, z_dim, u_dim)


class E2C(nn.Module):
    def __init__(self, obs_dim, z_dim, u_dim):
        super(E2C, self).__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.encoder = Encoder(3, z_dim)
        # self.encoder.apply(init_weights)
        self.decoder = Decoder(3, z_dim)
        # self.decoder.apply(init_weights)
        self.trans = PacmanTransition(z_dim=z_dim, u_dim=u_dim)
        # self.trans.apply(init_weights)

    def encode(self, x):
        """
        :param x:
        :return: mean and log variance of q(z | x)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        :param z:
        :return: bernoulli distribution p(x | z)
        """
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        """
        :param z_bar:
        :param q_z:
        :param u:
        :return: samples z_hat_next and Q(z_hat_next)
        """
        return self.trans(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)
        x_recon = self.decode(z)
        #print("z ", z.shape)
        #print("q_z", q_z.shape)
        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)

        mu_next, logvar_next = self.encode(x_next)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        return x_next_pred


class NormalDistribution:
    def __init__(self, mean, logvar, v=None, r=None, A=None):
        """
        :param mean: mu in the paper
        :param logvar: \Sigma in the paper
        :param v:
        :param r:
        if A is not None then covariance matrix = A \Sigma A^T, where A = I + v^T r
        else the covariance matrix is simply diag(logvar.exp())
        """
        self.mean = mean
        self.logvar = logvar
        self.v = v
        self.r = r

        sigma = torch.diag_embed(torch.exp(logvar))
        if A is None:
            self.cov = sigma
        else:
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))


    @staticmethod
    def KL_divergence(q_z_next_pred, q_z_next):
        """
        :param q_z_next_pred: q(z_{t+1} | z_bar_t, q_z_t, u_t) using the transition
        :param q_z_next: q(z_t+1 | x_t+1) using the encoder
        :return: KL divergence between two distributions
        """
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = torch.exp(q_z_next_pred.logvar)
        sigma_1 = torch.exp(q_z_next.logvar)
        v = q_z_next_pred.v
        r = q_z_next_pred.r
        k = float(q_z_next_pred.mean.size(1))

        sum = lambda x: torch.sum(x, dim=1)

        KL = 0.5 * torch.mean(sum((sigma_0 + 2*sigma_0*v*r) / sigma_1)
                              + sum(r.pow(2) * sigma_0) * sum(v.pow(2) / sigma_1)
                              + sum(torch.pow(mu_1-mu_0, 2) / sigma_1) - k
                              + 2 * (sum(q_z_next.logvar - q_z_next_pred.logvar) - torch.log(1 + sum(v*r)))
                              )
        return KL
