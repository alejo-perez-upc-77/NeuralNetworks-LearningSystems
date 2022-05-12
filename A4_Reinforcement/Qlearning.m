%% Initialization
%  Initialize the world, Q-table, and hyperparameters
worldnb = 2;
maxepochs = 1000;
alpha = 0.5; % learning rate 
gamma = 1.1; % discount factor 
eps = 0.6; % exploration rate
world = gwinit(worldnb);
% Q table: 
Q = zeros(world.ysize,world.xsize,4);



%% Training loop
%  Train the agent using the Q-learning algorithm.
for i = 1:maxepochs
    world = gwinit(worldnb);
    
    while true
        currentpos = gwstate().pos;
        [action, opt]= chooseaction(Q, currentpos(1), currentpos(2), [1,2,3,4] , [1,1,1,1], eps);
        newstate = gwaction(action);
        reward = getreward(100,-100, newstate);
        % update q value 
        % (getreward(100,-10000,newstate)
        Q(currentpos(1), currentpos(2), action) = ((1-alpha) * Q(currentpos(1), currentpos(2), action)) + (alpha * (newstate.feedback + (gamma * max(Q(newstate.pos(1), newstate.pos(2), :)))));
        if newstate.isterminal == 1
            break
        end
    end
    disp(i);
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

 world = gwinit(worldnb);
    
    while true
    gwdraw();
    currentpos = gwstate().pos;
    [action, opt]= chooseaction(Q, currentpos(1), currentpos(2), [1,2,3,4] , [1,1,1,1], 0);
    newstate = gwaction(action);
    pause(0.1)
    if newstate.isterminal == 1
        gwdraw();
        break
    end
    end

 
 %% plots 
P = getpolicy(Q);
gwdraw();
 gwdrawpolicy(P)

 
 figure(2)
 surf(getvalue(Q))
