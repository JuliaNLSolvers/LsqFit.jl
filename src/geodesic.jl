function vHv!(hessians,v,dir_deriv)
    tmp = similar(v) #v shouldn't be too large in general, so I kept it here
    vt = v'
    
    for i=1:length(dir_deriv)
        @views mul!(tmp, hessians[:,:,i], v)
        dir_deriv[i] = vt*tmp
    end
    
end 

