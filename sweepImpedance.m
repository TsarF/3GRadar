function z = sweepImpedance(arr,S,L,feedTrace,feedLocation,traceWidth,GroundPlaneLength,GroundPlaneWidth,freq)
    z = zeros(size(S));
    for i = 1:length(S)
        s = S(i)*L;
        truncatedCorners = createTruncatedCorners(arr,s);
    
        pcb = pcbStack(arr);
        pcb.Layers{1,1} = pcb.Layers{1,1} + feedTrace - truncatedCorners;
        pcb.FeedLocations = [0 feedLocation 1 3];
        pcb.ViaLocations = pcb.FeedLocations(1,:);
        pcb.FeedDiameter = traceWidth/2;
        pcb.ViaDiameter = pcb.FeedDiameter;
        pcb.Layers{1,2}.Length = GroundPlaneLength;
        pcb.Layers{1,2}.Width = GroundPlaneWidth;
        % manually mesh the antenna
        mesh(pcb,'MaxEdgeLength',1.2*traceWidth)
        z(i) = impedance(pcb,freq);
        mesh(pcb);
    end
end