## Results analysis of BLEVE-MLP

### Hidden = 1

- Training: p↓ train_loss↓
- p=0.2 is the best
- Batch size seems not a big problem
- batch size 512 seems best for val
- p↑ seems better for mape_test
- Best_mape_val top 5: batch=2048_p=0.2 (10.1%), batch=128_p=0.2, batch=256_p=0.3,batch=512_p=0.2,batch=512,p=0.1(10.87%)
- mape_test top 5: batch=1024_p=0.3 (10%),batch=512_p=0.3, batch=1024,p=0.1, batch=256_p=0.3,batch=512_p=0.2(10.88%)

## Hidden=2

- p↓ and neurons↑ can get better training loss
- batch size 512 seems best for val
- p=0 can get very good results but also potentially lead to bad val and test results (overfitting)
- Best_mape_val: top 5: n256_b128_p0.1(10.8%), n=128_b=1024_p=0.1, n256_b512_p0.1, n256_b512_p0(11.35%)
- Mape_test top 5: n256_b128_p0.1 (8.97%), n=128_b256_p0.4, n256_b2048_p0.1, n256_b512_p0.1(9.56%)

## Hidden = 3

- p↓ and neurons↑ can get better training loss
- batch512 still good, 
- p↓ good for val_loss, p=0, 0.1 is better for  val and test
- Best_mape_val: top 5: n256_b128_p0(11.27%), n256_b1024_p0.1, n256_b256_p0.1, n128_b1024_p0.1, n256_b512_p0.1(12.62%)
- Mape_test top 5: n256_b512_p0.1(10%), n256_b128_p0(10%), n256_b256_p0, n256_b1024_p0.1, n128_b256_p0(10.4%)

## impractical

- Output contains negative value, might need activation function for the output
- The distribution of target is lognormal, might need transformation

## Conclusion

Hidden=2, neurons=256, batchSize=512, p=0.1

SGD is better with mom=0.99

Softplus as the output activation function





