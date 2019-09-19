import folium

map = folium.Map(
    location=[-6.302403,106.652248],
    zoom_start=15
)
folium.Marker(
    [-6.302403,106.652248],
    popup='<b>Purwadhika</b>',
    tooltip="I'm here!"
).add_to(map)

map.save('0.html')