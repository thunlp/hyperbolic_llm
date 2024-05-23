sed -i "s/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/large_dp=0.1_attdp=0.1_lr=1e-3_step=30000/g" *
sed -i "s/bert_base.json/bert_large.json/g" *
sed -i "s/results\/rp/results\/large\/rp/g" *
