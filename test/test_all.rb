#!/usr/bin/ruby
if ARGV.count==0 then
    puts <<USAGE
    Compare the results from guru and mruby.
        Usage: test_all.rb [fname]
        where fname: xxx.rb or *.rb
    Place .mrb file in the same folder of .rb file
USAGE
    exit -1
end

MRUBY_PATH = "#{ENV['HOME']}/.rbenv/shims/"

$mruby = File.join(MRUBY_PATH, 'ruby')
$mrbc  = File.join(MRUBY_PATH, 'mrbc')
$guru  = '../Debug/guru'

[$mruby, $mrbc, $guru].each do |f|
    next if File.exists?(f)
    puts "'#{f}' not found."
    exit -2
end

exit

ARGV.each do |file|
  mrb_file = file.gsub('.rb', '.mrb')
  txt_file = file.gsub('.rb', '.txt')

  unless File.exists?(mrb_file) then
    `#{$mrbc_exe} -v -E #{file} > #{txt_file}`
  end

  if File.mtime(file) > File.mtime(mrb_file) then
    `#{$mrbc_exe} -v -E #{file} > #{txt_file}`
  end

  if File.exists?(mrb_file) then
    result_mruby = `#{$mruby_exe} #{file}`
    result_mrubyc = `#{$mrubyc_exe} #{mrb_file}`

    if result_mruby == result_mrubyc then
      puts "Success: #{file}"
    else
      puts "Fail: #{file}"
      puts "=====mruby====="
      puts result_mruby
      puts "=====mruby/c====="
      puts result_mrubyc
    end
  end
end
