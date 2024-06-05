function [Cjp,Cjs] = transfer_matrix_single_MAT(kz1,kz2,n1,n2,d)
% function [Cjp Cjs] = transfer_matrix_single_MAT(kz1,kz2,n1,n2,d)

[rp,rs,tp,ts] = fresnel_coeffs(kz1,kz2,n1,n2);

Cjs = cell(2,2);
Cjp = Cjs;

exp_plus_kz1d  = exp( 1i*kz1*d);
exp_minus_kz1d = exp(-1i*kz1*d);

Cjs{1,1} =  exp_minus_kz1d ./ ts;
Cjs{1,2} =  rs .* exp_minus_kz1d ./ ts;
Cjs{2,1} =  rs .* exp_plus_kz1d  ./ ts;
Cjs{2,2} =  exp_plus_kz1d ./ ts;

Cjp{1,1} =  exp_minus_kz1d ./ tp;
Cjp{1,2} =  rp .* exp_minus_kz1d ./ tp;
Cjp{2,1} =  rp .* exp_plus_kz1d  ./ tp;
Cjp{2,2} =  exp_plus_kz1d ./ tp;
