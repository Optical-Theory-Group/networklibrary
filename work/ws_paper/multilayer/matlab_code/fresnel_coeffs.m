function [rp,rs,tp,ts] = fresnel_coeffs(kz1,kz2,n1,n2)
    % function [rp,rs,tp,ts] = fresnel_coeffs(kz1,kz2,n1,n2)
    %   Calculates Fresnel reflection (rp,rs) and transmission (tp,ts)
    %   coefficients for s and p polarised light respectively for a single
    %   interface between media of refractive index n1 and n2. kz1 and kz2 are
    %   the axial component of the wavevector of the waves on either side of
    %   the interface and can be complex. 

    rp = -(n1.^2 .* kz2 - n2.^2 .* kz1)./(n1.^2 .* kz2 + n2.^2 .*kz1);
    rs = (kz1 - kz2)./(kz1 + kz2);
    tp = 2 .* n1 .* n2 .* kz1 ./ (n1.^2 .* kz2 + n2.^2 .* kz1);
    ts = 2 .*kz1 ./ (kz1 + kz2);
end