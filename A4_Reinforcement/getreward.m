function R = getreward( win,fail, newstate)
if newstate.isvalid == 1
    if newstate.isterminal == 1
        R = win;
    else 
        R = newstate.feedback;
    end
else
    R = fail;
end