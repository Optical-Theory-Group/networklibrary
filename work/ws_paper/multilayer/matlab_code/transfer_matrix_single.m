function [Cjp,Cjs] = transfer_matrix_single(kz1,kz2,n1,n2,d)
% function [Cjp Cjs] = transfer_matrix_single(kz1,kz2,n1,n2,d)

[rp,rs,tp,ts] = fresnel_coeffs(kz1,kz2,n1,n2);



Cjs= 1/ts  * [ exp(-1i*kz1*d)  ,rs * exp(-1i*kz1*d) ;
              rs*exp(1i*kz1*d) ,  exp(1i*kz1*d)];


Cjp = 1/tp * [ exp(-1i*kz1*d)  ,rp * exp(-1i*kz1*d) ;
              rp*exp(1i*kz1*d) ,  exp(1i*kz1*d)];
          