import torch
import numpy as np

def meta_test_il(ndp, ndp_opt, meta_loss_Network, num_epochs, batch_size, test_x, test_y, model_save_path):
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        inds = np.arange(test_x.shape[0])
        np.random.shuffle(inds)

        for ind in np.split(inds, len(inds) // batch_size):

            y_h = ndp.forward(test_x[ind], test_y[ind, 0, :])  # y is a 2D pose for all batches
            meta_input = torch.cat([test_x, y_h, test_y], dim=1)
            pred_task_loss = meta_loss_Network(meta_input).mean()
            ndp_opt.zero_grad()
            pred_task_loss.backward()
            ndp_opt.step()

        print("save")
        torch.save(ndp.state_dict(), model_save_path + '/model.pt')
