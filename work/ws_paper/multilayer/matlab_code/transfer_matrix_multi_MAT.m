function [Mp, Ms] = transfer_matrix_multi_MAT(KZJ,nJ, dJ)
% function [Mp, Ms] = transfer_matrix_multi_MAT(KZJ,nJ, dJ)

    Mp = cell(2);
    Mp{1,1} = ones(size(KZJ{1}));
    Mp{1,2} = 0*Mp{1,1};
    Mp{2,1} = 0*Mp{1,1};
    Mp{2,2} = Mp{1,1};

    Ms = Mp;
    Mp_temp = Mp;
    Ms_temp = Mp;
    

    for jj = 1:(numel(dJ)-1)
        [Cjp,Cjs] = transfer_matrix_single_MAT(KZJ{jj},KZJ{jj+1},nJ(jj),nJ(jj+1),dJ(jj));

        Mp_temp{1,1} = Cjp{1,1} .* Mp{1,1} + Cjp{2,1} .* Mp{1,2};
        Mp_temp{1,2} = Cjp{1,2} .* Mp{1,1} + Cjp{2,2} .* Mp{1,2};
        Mp_temp{2,1} = Cjp{1,1} .* Mp{2,1} + Cjp{2,1} .* Mp{2,2};
        Mp_temp{2,2} = Cjp{1,2} .* Mp{2,1} + Cjp{2,2} .* Mp{2,2};
        Mp = Mp_temp;
        
        Ms_temp{1,1} = Cjs{1,1} .* Ms{1,1} + Cjs{2,1} .* Ms{1,2};
        Ms_temp{1,2} = Cjs{1,2} .* Ms{1,1} + Cjs{2,2} .* Ms{1,2};
        Ms_temp{2,1} = Cjs{1,1} .* Ms{2,1} + Cjs{2,1} .* Ms{2,2};
        Ms_temp{2,2} = Cjs{1,2} .* Ms{2,1} + Cjs{2,2} .* Ms{2,2};
        Ms = Ms_temp;
    end

    exp_m_KZJ_d = exp(-1i*KZJ{end}*dJ(end));
    exp_p_KZJ_d = exp( 1i*KZJ{end}*dJ(end));

    Mp_temp{1,1} = exp_m_KZJ_d .* Mp{1,1};
    Mp_temp{1,2} = exp_p_KZJ_d .* Mp{1,2};
    Mp_temp{2,1} = exp_m_KZJ_d .* Mp{2,1};
    Mp_temp{2,2} = exp_p_KZJ_d .* Mp{2,2};
    Mp = Mp_temp;

    Ms_temp{1,1} = exp_m_KZJ_d .* Ms{1,1};
    Ms_temp{1,2} = exp_p_KZJ_d .* Ms{1,2};
    Ms_temp{2,1} = exp_m_KZJ_d .* Ms{2,1};
    Ms_temp{2,2} = exp_p_KZJ_d .* Ms{2,2};
    Ms = Ms_temp;
end