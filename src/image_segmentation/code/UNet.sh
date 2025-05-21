
for ((i=0;i<2;i++));do
	a='U_Net'
	python3 code/main.py --model_type=$a
	a='R2U_Net'
	python3 code/main.py --model_type=$a
	a='AttU_Net'
	python3 code/main.py --model_type=$a
	a='R2AttU_Net'
	python3 code/main.py --model_type=$a
	
done
