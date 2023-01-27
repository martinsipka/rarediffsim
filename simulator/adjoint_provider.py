import torch

TIMEFACTOR = 48.88821
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR
   
def get_adjoints(system, a, traj_tensor, end_indices, force_vjp, M, args):
    
    dt = args.dt / TIMEFACTOR
    gamma = args.gamma / PICOSEC2TIMEU
    a = a.to(args.device)    
    
    with torch.no_grad():
        a_dt = dt*args.save_steps

        b_i = torch.arange(traj_tensor.shape[0])

        adjoints = []
        testR = []
        R = traj_tensor[b_i, end_indices].detach().to(args.device)
        a = a*a_dt**2/M
        

        adjoints.append(a.detach())
        testR.append(R.detach())
        
        
        for i in range(0,args.backsteps):
            
            R = traj_tensor[b_i, end_indices-i-1].detach().to(args.device)

            vjp_a = force_vjp(R, a)
            a = a + a_dt**2 * vjp_a /M - a_dt*gamma * a

            adjoints.append(a.detach())
            testR.append(R.detach())
         
         
        adjoints = torch.stack(adjoints, axis=1)
        testR = torch.stack(testR, axis=1)


    return adjoints, testR
        
