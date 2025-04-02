python main.py --seed 1 --device cuda:0 --alg fedavg --dataset cifar10 --partition noniid-label-dir --beta 0.1 --global_iters 50 --local_iters 10 --n_parties 20 &
python main.py --seed 1 --device cuda:1 --alg metafed --dataset cifar10 --partition noniid-label-dir --beta 0.1 --global_iters 50 --local_iters 10 --n_parties 20 --is_val &
python main.py --seed 1 --device cuda:0 --alg feddng --dataset cifar10 --partition noniid-label-dir --beta 0.1 --global_iters 50 --local_iters 10 --n_parties 20 &
python main.py --seed 1 --device cuda:0 --alg fedprox --dataset cifar10 --partition noniid-label-dir --beta 0.1 --global_iters 50 --local_iters 10 --n_parties 20 &
python main.py --seed 1 --device cuda:1 --alg pfedgraph --dataset cifar10 --partition noniid-label-dir --beta 0.1 --global_iters 50 --local_iters 10 --n_parties 20 &

