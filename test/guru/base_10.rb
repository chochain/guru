# utf8
 
GUA = %w(地 雷 水 澤 山 火 風 天)
H64 = %w(
    天天乾　 地地坤　 水雷屯　 山水蒙　 風天小畜
).each do |u|
    hi, lo, v = u[0], u[1], u[2..3]
    hx = GUA.index(hi)*8 | GUA.index(lo)
    puts "#{hi}#{lo}:#{v}=>#{hx}"
end
    
    
