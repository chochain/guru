#!/usr/bin/ruby
if ARGV.count==0 then
    puts <<USAGE
    Compare the results from guru and ruby (or mruby).
        Usage: test_all.rb [fname]
        where fname: xxx.rb or *.rb
    Place .mrb file in the same folder of .rb file
USAGE
    exit -1
end

RUBY_PATH = "#{ENV['HOME']}/.rbenv/shims/"
GURU_PATH = "#{ENV['HOME']}/lib/guru/Debug/"

$ruby = File.join(RUBY_PATH, 'ruby')
$mrbc = File.join(RUBY_PATH, 'mrbc')
$guru = File.join(GURU_PATH, 'guru')

[$ruby, $mrbc, $guru].each do |f|
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
    
    rst_ruby = `#{$ruby} #{rb}`
    rst_guru = `#{$guru} #{mrb}`

    if rst_ruby == rst_guru
        puts "OK : #{rb}"
    else
        puts \
        "BAD: #{rb}\n"+
            "ruby #{'-'*30} ruby #{rb}\n#{rst_ruby}\n"+
            "guru #{'+'*30} guru #{rb}\n#{rst_guru}\n"
    end
end
