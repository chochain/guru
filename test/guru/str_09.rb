# coding: utf-8
# string#include?
    
A = %w(天天乾　 地地坤　 水雷屯)
S = %(天天 地地 天雷)

A.each do |a|
    x = a[0..1]
    p x
    p S.include?(x)
end 
