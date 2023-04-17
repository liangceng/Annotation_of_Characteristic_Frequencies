function [solutions, errors] = fre_combination_ILP(f_target, f_can,varargin)
% find optimal integer linear combination of f_target on f_can with ILP
% email:tomoroling@gmail.com
%{
%%input:
f_target: target frequency, a number
f_can: frequecies candidates, a 1Xn array
N_max
epsilon: 误差范围
weight_Coef
%%output:
solutions
errors
%}
%% initialization for optional inputs
weight_Coef=ones(size(f_can));
epsilon=1e-1;
N_max=2;% default value of N_max
N_min=-2;
Disp=1; %default display mode
x0=[];
if nargin>=8
    x0=varargin{6};
end
if nargin>=7
    Disp=varargin{5};
end
if nargin>=6
    weight_Coef=varargin{4};
end
if nargin>=5
    N_min=varargin{3};
end
if nargin>=4
    N_max=varargin{2};
end
if nargin>=3
    epsilon=varargin{1};
end
N_can=length(f_can);
if max(size(N_max))==1
    N_max=ones(N_can,1)*N_max;
end
if max(size(N_min))==1
    N_min=ones(N_can,1)*N_min;
end
if size(f_can,1)>1
    f_can=f_can';
end
if size(weight_Coef,1)>1
    weight_Coef=weight_Coef';
end
if size(N_max,1)>1
    N_max=N_max';
end
if size(N_min,1)>1
    N_min=N_min';
end
% if size(x0,1)>1
%     x0=x0';
% end
% if length(x0)==1
%     x0=x0*ones(1,N_can);
% end
n = length(f_can);
solutions = [];
errors=[];
for i=1:length(f_target)
    [solution, error] = L1norm_ILP(f_target(i), f_can, N_max, N_min, epsilon, weight_Coef,x0);
    solutions(i,:)=solution;
    errors(i,:)=error;
end
solutions=int32(solutions);
end

function [solutions, error] = L1norm_ILP(g, h, xmax, xmin, eps, weight_Coef,x0)
%find x, min(x) |x| s.t. -eps<=h'x-g<=eps,  -xmax<=x<=xmax
% linearized version is min(y,z) y+z, s.t. y-z=x, y>=0,z>=0
    n = length(h);
    if ~isempty(x0)
        x0=[x0 zeros(1,n)];
    end
    f = [1./weight_Coef'; 1./weight_Coef'];
    % A = [-diag(h), -eye(n); diag(h), -eye(n)];
    % A1=[h,-h];
    % A2=[-h,h];
    % A3=[ones(1,n), -ones(1,n)];
    % A4=[-ones(1,n), ones(1,n)];
    A=[h,-h;...%h'x-g<=eps
        -h,h;...%h'x-g>=-eps
        [eye(n), -eye(n)];...%x<=xmax
        [-eye(n), eye(n)];];%x>=-xmax
    b=[g+eps,-g+eps,xmax,-xmin];
    b_lb = zeros(2*n,1);
    b_ub = inf*ones(2*n,1);
    intcon = 1:2*n;
    options = optimoptions('intlinprog','Display','none','MaxTime',5,'OutputFcn','savemilpsolutions');
    [x,~,exitflag] = intlinprog(f,intcon,A,b,[],[],b_lb,b_ub,x0,options);
    solutions=zeros(1,n);
    error=zeros(1,n);
    if exitflag == 1
        x = x(1:n) - x(n+1:end);
        error = abs(x'*h' - g);
        if error <= eps
            solutions = x';
        end
    end
end