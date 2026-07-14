function width = traceThickness(z,d)
    A = z/60*sqrt((d.EpsilonR+1)/2)+(d.EpsilonR-1)/(d.EpsilonR+1)*(0.23+0.11/d.EpsilonR);
    width = d.Thickness*8*exp(A)/(exp(2*A)-2);
end