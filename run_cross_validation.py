import torch
import numpy as np
from config import parser
import time
from evaluator import Evaluator
import util
from optimizer import StochasticGD

args = parser.parse_args()


def cross_validation():
    device = torch.device(args.device)
    mklrs, jsds, earths, maes, rmses = [],[],[],[],[]
    for fold in range(5):
        args.fold = fold
        adj_mx = util.load_adj(args.adjdata)
        dataloader = util.load_dataset(args)

        adj = [torch.tensor(i).to(device) for i in adj_mx]

        engine = StochasticGD(args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, adj)

        print("start train...", flush=True)
        his_loss = []
        val_time = []
        train_time = []
        for i in range(1, args.epochs + 1):
            train_loss = []
            t1 = time.time()
            for iter, (x, y, w, c) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)  # Batch_size, seq_length, nodes, histograms
                trainy = torch.Tensor(y).to(device)
                trainw = torch.Tensor(w).to(device)
                train_tmp_ctx = torch.Tensor(c[0]).to(device)
                train_spat_ctx = torch.Tensor(c[1]).to(device)
                metrics = engine.train(trainx, trainy, trainw, [train_tmp_ctx, train_spat_ctx])

                train_loss.append(metrics)
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, KL: {:.4f}'
                    print(log.format(iter, train_loss[-1]), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            # validation
            valid_loss = []

            s1 = time.time()
            for iter, (x, y, w, c) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testy = torch.Tensor(y).to(device)
                testw = torch.Tensor(w).to(device)
                test_tmp_ctx = torch.Tensor(c[0]).to(device)
                test_spat_ctx = torch.Tensor(c[1]).to(device)
                metrics = engine.eval(testx, testy, testw, [test_tmp_ctx, test_spat_ctx])
                valid_loss.append(metrics)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)

            mvalid_loss = np.mean(valid_loss)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mvalid_loss, (t2 - t1)), flush=True)
            torch.save(engine.model.state_dict(),
                       args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        # test
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(
            torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

        outputs = []
        really = []
        weights = []

        for iter, (x, y, w, c) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testw = torch.Tensor(w).to(device)
            test_tmp_ctx = torch.Tensor(c[0]).to(device)
            test_spat_ctx = torch.Tensor(c[1]).to(device)

            with torch.no_grad():
                engine.model.is_test = True
                preds = engine.model(testx, [test_tmp_ctx, test_spat_ctx])

            outputs.append(preds)
            really.append(testy)
            weights.append(testw)

        yhat = torch.cat(outputs, dim=0)
        ytrue = torch.cat(really, dim=0)
        weights = torch.cat(weights, dim=0)

        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

        evaluator = Evaluator(args)

        mklr = evaluator.mklr(ytrue, yhat, weights)
        jsd = evaluator.jesen_shannon_divergence(ytrue, yhat, weights)
        wass = evaluator.weighted_emd(ytrue, yhat, weights)
        h_metrics = evaluator.hist_metrics(ytrue, yhat, weights)

        log = 'Evaluate best model on test data fold {}, Test MKLR: {:.4f}, Test JSD: {:.4f}, Test Earth Distance: {:.4f}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(fold, mklr, jsd, wass, h_metrics[0], h_metrics[1]))
        mklrs.append(mklr)
        jsds.append(jsd)
        earths.append(wass)
        maes.append(h_metrics[0])
        rmses.append(h_metrics[1])

    mklrs = torch.Tensor(mklrs).to('cpu')
    jsds = torch.Tensor(jsds).to('cpu')
    earths = torch.Tensor(earths).to('cpu')
    maes = torch.Tensor(maes).to('cpu')
    rmses = torch.Tensor(rmses).to('cpu')
    log = 'On average Test MKLR: {:.4f}, Test JSD: {:.4f}, Test ED: {:.4f}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(torch.mean(mklrs), torch.mean(jsds), torch.mean(earths), torch.mean(maes), torch.mean(rmses)))
    print(args.save + "_exp_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    cross_validation()
    print('Done!')