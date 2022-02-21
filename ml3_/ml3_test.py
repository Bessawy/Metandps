import torch
import numpy as np
import matplotlib.pyplot as plt

def meta_test_il(ndp, meta_loss_Network, num_epochs, batch_size, X_train, Y_train, X_test, Y_test,
                 model_save_path):

    ndp_opt = torch.optim.Adam(ndp.parameters(), lr=1e-3)
    meta_loss_Network.load_state_dict(torch.load('ml3_loss_sminst.pt'), strict=False)
    ndp.reset()

    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        inds = np.arange(X_train.shape[0])
        np.random.shuffle(inds)

        for ind in np.split(inds, len(inds) // batch_size):
            y_h = ndp.forward(X_train[ind], Y_train[ind, 0, :])  # y is a 2D pose for all batches
            pred_task_loss = meta_loss_Network(y_h, Y_train[ind])
            ndp_opt.zero_grad()
            pred_task_loss.backward()
            ndp_opt.step()
            #print(torch.mean((y_h - Y_train[ind]) ** 2))

        torch.save(ndp.state_dict(), model_save_path + '/model.pt')


        if epoch % 20 == 0:

            x_test = X_test[np.arange(100)]
            y_test = Y_test[np.arange(100)]
            y_htest = ndp.forward(x_test, y_test[:, 0, :])
            for j in range(18):
                plt.figure(figsize=(8, 8))
                plt.plot(0.667 * y_h[j, :, 0].detach().cpu().numpy(), -0.667 * y_h[j, :, 1].detach().cpu().numpy(),
                         c='r', linewidth=5)
                plt.axis('off')
                plt.savefig(model_save_path + '/train_img_' + str(j) + '.png')

                plt.figure(figsize=(8, 8))
                img = X_train[ind][j].cpu().numpy() * 255
                img = np.asarray(img * 255, dtype=np.uint8)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(model_save_path + '/ground_train_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

                plt.figure(figsize=(8, 8))
                plt.plot(0.667 * y_htest[j, :, 0].detach().cpu().numpy(),
                         -0.667 * y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
                plt.axis('off')
                plt.savefig(model_save_path + '/test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

                plt.figure(figsize=(8, 8))
                img = X_test[j].cpu().numpy() * 255
                img = np.asarray(img * 255, dtype=np.uint8)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(model_save_path + '/ground_test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            test = ((y_htest - y_test) ** 2).mean(1).mean(1)
            print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))