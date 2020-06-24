# raise exception
begin
    raise "err"
rescue => e
    p ":#{e}"
end 
begin
    raise StandardError.new "stderr"
rescue => e
    p ":#{e}"
end 
