function truncatedCorners = createTruncatedCorners(arr,s)   
    x11 = -arr.ColumnSpacing/2 + arr.Element.Length/2;
    y11 = arr.RowSpacing/2 + arr.Element.Width/2;
    x12 = -arr.ColumnSpacing/2 - arr.Element.Length/2;
    y12 = arr.RowSpacing/2 - arr.Element.Width/2;
    s1 = antenna.Polygon(Vertices=[x11-s y11 0; x11 y11-s 0; x11 y11 0]);
    s2 = antenna.Polygon(Vertices=[x12+s y12 0; x12 y12+s 0; x12 y12 0]);

    x21 = x11;
    y21 = -arr.RowSpacing/2+arr.Element.Width/2;
    x22 = x12;
    y22 = -arr.RowSpacing/2-arr.Element.Width/2;
    s3 = antenna.Polygon(Vertices=[x21-s y21 0; x21 y21-s 0; x21 y21 0]);
    s4 = antenna.Polygon(Vertices=[x22+s y22 0; x22 y22+s 0; x22 y22 0]);

    x31 = arr.ColumnSpacing/2 + arr.Element.Length/2;
    y31 = y11;
    x32 = arr.ColumnSpacing/2 - arr.Element.Length/2;
    y32 = y12;
    s5 = antenna.Polygon(Vertices=[x31-s y31 0; x31 y31-s 0; x31 y31 0]);
    s6 = antenna.Polygon(Vertices=[x32+s y32 0; x32 y32+s 0; x32 y32 0]);

    x41 = x31;
    y41 = y21;
    x42 = x32;
    y42 = y22;
    s7 = antenna.Polygon(Vertices=[x41-s y41 0; x41 y41-s 0; x41 y41 0]);
    s8 = antenna.Polygon(Vertices=[x42+s y42 0; x42 y42+s 0; x42 y42 0]);

    truncatedCorners = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
end