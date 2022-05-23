import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from util.visualizer import Visualizer


from models import networks

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)  

    val_opt = opt
    val_opt.phase = 'test'      # Select data from test folder for validation
    val_dataset = create_dataset(val_opt)
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    dataset_size = len(val_dataset)    # get the number of images in the dataset.
    print('The number of validation images = %d' % dataset_size)

    domain_disc = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids, opt.q, opt.is_residual, not opt.no_bias)
    
    loss_func = networks.GANLoss(opt)    # Use standard GAN loss for training
    optimizer = torch.optim.Adam(domain_disc.parameters(), lr=opt.lr)
    best_val_loss = 1000

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        
        train_loss = []
        val_loss = []

        domain_disc.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch

            outputA = domain_disc(data['A'].to(device), False)     # False for source distribution
            outputB = domain_disc(data['B'].to(device), True)      # True for target distribution

            loss = loss_func(outputA) + loss_func(outputB)
            train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        domain_disc.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset):  # inner loop within one epoch

                outputA = domain_disc(data['A'], False)     # False for source distribution
                outputB = domain_disc(data['B'], True)      # True for target distribution

                loss = loss_func(outputA) + loss_func(outputB)
                val_loss.append(loss.item())

        train_loss = torch.mean(train_loss)
        val_loss = torch.mean(val_loss)

        if val_loss < best_val_loss:              # cache our model every <save_epoch_freq> epochs
            print(f'New best validation loss {val_loss}, saving the model at the end of epoch {epoch}.')
            torch.save(domain_disc.cpu().state_dict(), "domain_discriminator_model.pth.tar")
            best_val_loss = val_loss

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
