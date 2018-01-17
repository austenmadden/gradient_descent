require 'csv'

class GradientDescent
  def execute(iterations: 1000, rate: 0.0001, b: 0.0, m: 0.0)
    points = { x: [], y: [] }
    CSV.foreach('data.csv') do |row|
      points[:x] << row[0].to_f
      points[:y] << row[1].to_f
    end

    init_e = compute_error(b, m, points)
    puts "Starting at b = #{b}, m = #{m}, error = #{init_e}"
    b, m = gradient_descent(points, b, m, rate, iterations)
    e = compute_error(b, m, points)
    puts "After #{iterations} iterations, b = #{b}, m = #{m}, error = #{e}"
  end

  private

  def gradient_descent(points, b, m, rate, iterations)
    iterations.times do
      b, m = step_gradient(points, b, m, rate)
    end

    [b, m]
  end

  def step_gradient(points, b, m, rate)
    b_gradient, m_gradient = 0, 0
    length = points[:x].size

    length.times do |i|
      x = points[:x].at(i)
      y = points[:y].at(i)

      b_gradient += -(2.0/length) * (y - ((m * x) + b))
      m_gradient += -(2.0/length) * x * (y - ((m * x) + b))
    end

    new_b = b - (rate * b_gradient)
    new_m = m - (rate * m_gradient)

    [new_b, new_m]
  end

  def compute_error(b, m, points)
    total_error = 0

    points[:x].size.times do |i|
      x = points[:x].at(i)
      y = points[:y].at(i)

      total_error += (y - (m * x + b)) ** 2
    end

    total_error/points[:x].size
  end
end

GradientDescent.new.execute
