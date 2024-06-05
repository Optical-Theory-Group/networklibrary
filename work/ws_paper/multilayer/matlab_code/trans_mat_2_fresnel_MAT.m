function [rp,rs,tp,ts] = trans_mat_2_fresnel_MAT(Mp,Ms)

rp = Mp{2,1} ./ Mp{1,1};
rs = Ms{2,1} ./ Ms{1,1};

tp = 1 ./ Mp{1,1};
ts = 1 ./ Ms{1,1};