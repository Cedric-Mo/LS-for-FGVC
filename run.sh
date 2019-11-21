python main.py --dataset cubbirds --nparts 1 --gamma1 0 --gamma2 0 --device 0,1
python main.py --dataset cubbirds --nparts 1 --gamma1 0 --gamma2 0.8 --device 0,1
python main.py --dataset cubbirds --osme --nparts 2 --gamma1 0 --gamma2 0 --device 0,1
python main.py --dataset cubbirds --osme --nparts 2 --gamma1 1 --gamma2 0 --device 0,1
python main.py --dataset cubbirds --osme --nparts 2 --gamma1 1 --gamma2 0.8 --device 0,1

python main2.py --dataset stcars --nparts 1 --gamma1 0 --gamma2 0 --device 0,1
python main2.py --dataset stcars --nparts 1 --gamma1 0 --gamma2 0.65 --device 0,1
python main2.py --dataset stcars --osme --nparts 2 --gamma1 0 --gamma2 0 --device 0,1
python main2.py --dataset stcars --osme --nparts 2 --gamma1 1 --gamma2 0 --device 0,1
python main2.py --dataset stcars --osme --nparts 2 --gamma1 1 --gamma2 0.65 --device 0,1

python main2.py --dataset stdogs --nparts 1 --gamma1 0 --gamma2 0 --milestones 8 30 --lr 0.001 --device 0,1
python main2.py --dataset stdogs --nparts 1 --gamma1 0 --gamma2 0.65 --milestones 8 30 --lr 0.001 --device 0,1
python main2.py --dataset stdogs --osme --nparts 2 --gamma1 1 --gamma2 0 --milestones 8 30 --lr 0.001 --device 0,1
python main2.py --dataset stdogs --osme --nparts 2 --gamma1 0 --gamma2 0 --milestones 8 30 --lr 0.001 --device 0,1
python main2.py --dataset stdogs --osme --nparts 2 --gamma1 1 --gamma2 0.65 --milestones 8 30 --lr 0.001 --device 0,1

python main2.py --dataset vggaircraft --nparts 1 --gamma1 0 --gamma2 0 --device 0,1
python main2.py --dataset vggaircraft --nparts 1 --gamma1 0 --gamma2 0.65 --device 0,1
python main2.py --dataset vggaircraft --osme --nparts 2 --gamma1 0 --gamma2 0 --device 0,1
python main2.py --dataset vggaircraft --osme --nparts 2 --gamma1 1 --gamma2 0 --device 0,1
python main2.py --dataset vggaircraft --osme --nparts 2 --gamma1 1 --gamma2 0.65 --device 0,1

python main.py --dataset nabirds --nparts 1 --gamma1 0 --gamma2 0 --device 0,1
python main.py --dataset nabirds --nparts 1 --gamma1 0 --gamma2 0.65 --device 0,1
python main.py --dataset nabirds --osme --nparts 2 --gamma1 0 --gamma2 0 --device 0,1
python main.py --dataset nabirds --osme --nparts 2 --gamma1 1 --gamma2 0 --device 0,1
python main.py --dataset nabirds --osme --nparts 2 --gamma1 1 --gamma2 0.65 --device 0,1
