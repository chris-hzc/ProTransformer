## Robust Attention

Core code: `transformers/models/bert/modeling_bert.py`

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
