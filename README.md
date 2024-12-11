# ProTransformer: Robustify Transformers via Plug-and-Play Paradigm


This repository provides the code, data, and results for the paper: ProTransformer: Robustify Transformers via Plug-and-Play Paradigm [[arXiv](https://arxiv.org/abs/2410.23182)].

<p float="left">
  <img src="./figures/textattack.png" width="61.5%" />
  <img src="./figures/protransformer.png" width="35%" /> 
</p>


## Core Codes

## Robust Attention

Example core code: `transformers/models/bert/modeling_bert.py`

```
class RobustSum(nn.Module):
    def __init__(self, L=3, norm="L2", epsilon =1e-2, gamma=4.0, t=1.0, delta=4.0):
        super().__init__()
        self.L=L
        self.norm=norm
        self.epsilon=epsilon
        self.gamma=gamma
        self.t=t
        self.delta=delta

    def forward(self, A, V):

        M = torch.matmul(A, V)

        
        if self.norm == 'L2':
            return M


        for _ in range(self.L):
            dist = torch.cdist(M.detach(),V.detach())
            #dist = torch.cdist(M,V)
            
            if self.norm == 'L1':
                w = 1/(dist+self.epsilon)
                
            elif  self.norm == 'MCP':
                w = 1/(dist + self.epsilon) - 1/self.gamma
                w[w<self.epsilon]=self.epsilon
                
            elif self.norm == 'Huber':
                w = self.delta/(dist + self.epsilon)
                w[w>1.0] = 1.0
            
            ww = w * A
            
            ww_norm = torch.nn.functional.normalize(ww,p=1,dim=-1)
            
            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm,V)
            torch.cuda.empty_cache()
            
        return M
```

## Requirements

This project is built upon [Python 3.10](https://www.python.org).


## Reproducing Results

### Classical Text Attack
```bash
python main.py --backbone='bert' --norm='MCP' --gamma=4.0 --epsilon=1e-2 --L=3 --data='ag-news' --attack='tf'
```


## Experimental Results



<p float="left">
  <img src="./figures/results.png" width="100%" />
  <!-- <img src="./figures/instr-108.jpg" width="54%" />  -->
</p>

## Citation

If you find our work helpful, please consider citing it as
```
@article{hou2024protransformer,
  title={ProTransformer: Robustify Transformers via Plug-and-Play Paradigm},
  author={Hou, Zhichao and Gao, Weizhi and Shen, Yuchen and Wang, Feiyi and Liu, Xiaorui},
  journal={arXiv preprint arXiv:2410.23182},
  year={2024}
}
```





