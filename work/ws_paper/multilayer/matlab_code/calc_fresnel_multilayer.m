function [R1NS,R1NP,T1NS,T1NP] = calc_fresnel_multilayer(nJ,dJ,k0,KR)
% function [R1NS,R1NP,T1NS,T1NP] = calc_gen_fresnel_multilayer(nJ,dJ,KZ)
%
% Inputs
%   nJ  = vector of refractive indices for layer structure [n1,...nj,...nN]
%   dJ  = vector of layer thicknesses [0,d1,d2,...dn-1,0] where di = z_i+1 - zi
%   KZ  = cell array of KZ component in each layer
%
% Outputs
%   R1NS,R1NP  = cell array of generalised reflection coefficients going
%                from layer 1 to N.
%   T1NS,T1NP  = cell array of generalised transmission coefficients going
%                from layer 1 to N.

    %% Initialise necessary cell arrays
    Nlayers = length(dJ);
    KZ = cell(1,Nlayers);
    for layer_ind = 1:Nlayers
        KZ{layer_ind} = sqrt(nJ(layer_ind)^2 * k0^2 - KR.^2);
    end

    [MP0, MS0] = transfer_matrix_multi_MAT(KZ,nJ,dJ);
    [R1NP,R1NS,T1NP,T1NS] = trans_mat_2_fresnel_MAT(MP0,MS0);

    % remove problematic NaNs and Infs
    R1NP(~isfinite(R1NP)) = 0;
    R1NS(~isfinite(R1NS)) = 0;
    T1NP(~isfinite(R1NP)) = 0;
    T1NS(~isfinite(R1NS)) = 0;
end
        
