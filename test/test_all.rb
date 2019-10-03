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
GURU_PATH = "#{ENV['HOME']}/lib/guru/Debug/"

$mruby = File.join(MRUBY_PATH, 'ruby')
$mrbc  = File.join(MRUBY_PATH, 'mrbc')
$guru  = File.join(GURU_PATH,  'guru')

[$mruby, $mrbc, $guru].each do |f|
    next if File.exists?(f)
    puts "'#{f}' not found."
    exit -2
end

ARGV.each do |rb|
    mrb = rb.gsub('.rb', '.mrb')
    txt = rb.gsub('.rb', '.txt')

    unless File.exists?(mrb) &&
            File.mtime(rb) < File.mtime(mrb)
        puts "#{$mrbc} --verbose -o #{mrb} #{rb} > #{txt}"
        `#{$mrbc} --verbose -o #{mrb} #{rb} > #{txt}`
    end

    next unless File.exists?(mrb)
    
    rst_mruby = `#{$mruby} #{rb}`
    rst_guru  = `#{$guru}  #{mrb}`

    if rst_mruby == rst_guru
        puts "OK : #{rb}"
    else
        puts \
        "BAD: #{rb}\n"+
            "mruby #{'-'*30} mruby #{rb}\n#{rst_mruby}\n"+
            "guru  #{'+'*30} guru  #{rb}\n#{rst_guru}\n"
    end
end
