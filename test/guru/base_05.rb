# for loop
#   Range object
#   LAMBDA, SENDB
#   SETUPVAR, GETUPVAR

for i in 1..5 do
    puts i
end
puts i

[ 1, 2, "X" ].each do |j|
    puts j
end
begin
    puts j
rescue => e
    puts "err"
end
    
